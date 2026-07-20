"""LLM abstraction for memory systems.

Thin wrapper around litellm.completion with GPT-OSS harmony parsing.
"""

import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

from litellm import completion as litellm_completion
from litellm import completion_cost, token_counter
from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(
    os.path.expanduser(
        os.environ.get(
            "SPREADSHEET_BENCH_LLM_CACHE_DIR",
            "~/.cache/spreadsheet-bench/litellm",
        )
    )
)
CACHE_VERSION = 1
KNOWN_PROVIDER_PREFIXES = (
    "anthropic/",
    "azure/",
    "bedrock/",
    "cohere/",
    "gemini/",
    "groq/",
    "ollama/",
    "openai/",
    "openrouter/",
    "together_ai/",
    "togethercomputer/",
    "vertex_ai/",
    "xai/",
)

MAX_PROMPT_CHARS = 224_000

_HARMONY_ENC = None


def _get_harmony_enc():
    """Lazy-load harmony encoder to avoid import overhead when not using GPT-OSS."""
    global _HARMONY_ENC
    if _HARMONY_ENC is None:
        _HARMONY_ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _HARMONY_ENC


def parse_harmony_response(raw_content: str) -> str:
    """Parse GPT-OSS harmony format, extracting final channel content.

    Returns the 'final' channel content, or raw content if parsing fails.
    """
    enc = _get_harmony_enc()
    try:
        tokens = enc.encode(raw_content, allowed_special="all")
        parsed = enc.parse_messages_from_completion_tokens(
            tokens, role=Role.ASSISTANT, strict=False
        )

        for msg in parsed:
            if msg.channel == "final":
                return "".join(c.text for c in msg.content if hasattr(c, "text"))

        if parsed:
            return "".join(c.text for c in parsed[-1].content if hasattr(c, "text"))
    except Exception:
        pass

    return raw_content


def _is_retryable(exc: Exception) -> bool:
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(exc, "status", None)
    )
    if status == 429:
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "429",
            "rate limit",
            "too many requests",
            "retry in",
            "timed out",
            "timeout",
        )
    )


def _extract_content(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "".join(parts)
    return ""


class LLMCallable(Protocol):
    """Protocol for LLM call functions."""

    def __call__(self, prompt: str) -> str: ...


class ProviderLLM:
    """Thin shim that provides the old provider interface on top of litellm."""

    def __init__(
        self,
        model: str,
        max_concurrent: int = 4,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_concurrent = max_concurrent
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self._usage_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __exit__(self, *exc):
        self._executor.shutdown(wait=True)
        return False

    def _normalized_model(self) -> str:
        if self.api_base and not self.model.startswith(KNOWN_PROVIDER_PREFIXES):
            return f"openai/{self.model}"
        return self.model

    def _cache_path(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> Path:
        payload = {
            "version": CACHE_VERSION,
            "model": self._normalized_model(),
            "api_base": self.api_base,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "kwargs": kwargs,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return CACHE_DIR / f"{digest}.json"

    def _load_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            with self._cache_lock:
                tmp.write_text(json.dumps(payload))
                tmp.replace(path)
        except OSError:
            pass

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=32) + wait_random(1, 5),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call_completion(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs = dict(kwargs)
        call_kwargs["timeout"] = 600.0
        if self.api_base:
            call_kwargs["base_url"] = self.api_base
            call_kwargs.setdefault("api_key", self.api_key or "local")
        elif self.api_key is not None:
            call_kwargs["api_key"] = self.api_key

        model = self._normalized_model()
        response = litellm_completion(model=model, messages=messages, **call_kwargs)
        content = _extract_content(response)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is None:
            try:
                input_tokens = token_counter(model=model, messages=messages)
            except Exception:
                input_tokens = 0
        if output_tokens is None:
            try:
                output_tokens = token_counter(model=model, text=content)
            except Exception:
                output_tokens = 0

        if self.api_base:
            cost = 0.0
        else:
            try:
                cost = float(completion_cost(completion_response=response) or 0.0)
            except Exception:
                cost = 0.0

        return {
            "content": content,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "cost": cost,
        }

    def _generate_one(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> str:
        cache_path = self._cache_path(prompt, system_prompt, kwargs)
        cached = self._load_cache(cache_path)
        if cached is not None:
            return cached["content"]

        result = self._call_completion(prompt, system_prompt, kwargs)
        self._save_cache(cache_path, result)

        with self._usage_lock:
            self.total_input_tokens += result["input_tokens"]
            self.total_output_tokens += result["output_tokens"]
            self.total_cost += result["cost"]

        return result["content"]

    def generate(self, prompts, system_prompt: str | None = None, **kwargs):
        if isinstance(prompts, str):
            return [[self._generate_one(prompts, system_prompt, kwargs)]]

        prompts = list(prompts)
        if not prompts:
            return []

        results = [None] * len(prompts)
        futures = {
            self._executor.submit(
                self._generate_one, prompt, system_prompt, kwargs
            ): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
        return [[content] for content in results]

    def _cache_path_msgs(self, messages: list[dict], kwargs: dict[str, Any]) -> Path:
        payload = {
            "version": CACHE_VERSION,
            "model": self._normalized_model(),
            "api_base": self.api_base,
            "messages": messages,
            "kwargs": kwargs,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return CACHE_DIR / f"chat_{digest}.json"

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=32) + wait_random(1, 5),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call_chat(
        self, messages: list[dict], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        call_kwargs = dict(kwargs)
        call_kwargs["timeout"] = 600.0
        if self.api_base:
            call_kwargs["base_url"] = self.api_base
            call_kwargs.setdefault("api_key", self.api_key or "local")
        elif self.api_key is not None:
            call_kwargs["api_key"] = self.api_key

        model = self._normalized_model()
        response = litellm_completion(model=model, messages=messages, **call_kwargs)
        content = _extract_content(response)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is None:
            try:
                input_tokens = token_counter(model=model, messages=messages)
            except Exception:
                input_tokens = 0
        if output_tokens is None:
            try:
                output_tokens = token_counter(model=model, text=content)
            except Exception:
                output_tokens = 0

        if self.api_base:
            cost = 0.0
        else:
            try:
                cost = float(completion_cost(completion_response=response) or 0.0)
            except Exception:
                cost = 0.0

        return {
            "content": content,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "cost": cost,
        }

    def chat(
        self, messages: list[dict], use_cache: bool = True, **kwargs
    ) -> str:
        """Multi-turn chat completion over an explicit messages list."""
        cache_path = self._cache_path_msgs(messages, kwargs)
        if use_cache:
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached["content"]

        result = self._call_chat(messages, kwargs)
        if use_cache:
            self._save_cache(cache_path, result)

        with self._usage_lock:
            self.total_input_tokens += result["input_tokens"]
            self.total_output_tokens += result["output_tokens"]
            self.total_cost += result["cost"]

        return result["content"]


class LLM:
    """LLM caller backed by litellm. Handles caching, batching, retries."""

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-oss-120b",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 16384,
        max_workers: int = 32,
    ):
        self.model = model
        self.is_gpt_oss = "gpt-oss" in model.lower()

        if temperature is not None:
            self.temperature = temperature
        elif "gpt-5" in model or self.is_gpt_oss:
            self.temperature = 1.0
        else:
            self.temperature = 0.0

        if api_base is not None and not self.is_gpt_oss and max_tokens > 4096:
            max_tokens = 4096
        self.max_tokens = max_tokens

        self._model_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        self._system_prompt = "Reasoning: medium" if self.is_gpt_oss else None

        if api_base is None and max_workers > 4:
            max_workers = 4
        self._provider = ProviderLLM(
            model=model,
            max_concurrent=max_workers,
            api_key=api_key,
            api_base=api_base,
        )

        self._usage_lock = threading.Lock()
        self.total_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if hasattr(self._provider, "__exit__"):
            self._provider.__exit__(*exc)
        return False

    @property
    def total_input_tokens(self):
        return self._provider.total_input_tokens

    @property
    def total_output_tokens(self):
        return self._provider.total_output_tokens

    @property
    def total_cost(self):
        return self._provider.total_cost

    def _truncate(self, prompt: str) -> str:
        if len(prompt) > MAX_PROMPT_CHARS:
            original_len = len(prompt)
            half = MAX_PROMPT_CHARS // 2
            prompt = prompt[:half] + "\n\n... [TRUNCATED] ...\n\n" + prompt[-half:]
            logger.warning(
                "Prompt truncated: %d -> %d chars (limit %d)",
                original_len,
                len(prompt),
                MAX_PROMPT_CHARS,
            )
        return prompt

    def __call__(self, prompt: str) -> str:
        """Single prompt call."""
        prompt = self._truncate(prompt)

        results = self._provider.generate(
            prompt, system_prompt=self._system_prompt, **self._model_kwargs
        )
        content = results[0][0]

        with self._usage_lock:
            self.total_calls += 1

        if self.is_gpt_oss:
            content = parse_harmony_response(content)
        return content

    def chat(self, messages: list[dict], use_cache: bool = True) -> str:
        """Multi-turn chat over a messages list (for agent loops).

        Each message is a dict {"role": "system"|"user"|"assistant", "content": str}.
        For GPT-OSS the harmony reasoning directive is injected into the leading
        system message and the response is parsed down to its final channel.
        """
        msgs = [dict(m) for m in messages]
        if self.is_gpt_oss:
            if msgs and msgs[0].get("role") == "system":
                if not str(msgs[0].get("content", "")).startswith("Reasoning:"):
                    msgs[0]["content"] = "Reasoning: medium\n\n" + msgs[0]["content"]
            else:
                msgs.insert(0, {"role": "system", "content": "Reasoning: medium"})

        for m in msgs:
            if isinstance(m.get("content"), str):
                m["content"] = self._truncate(m["content"])

        content = self._provider.chat(msgs, use_cache=use_cache, **self._model_kwargs)

        with self._usage_lock:
            self.total_calls += 1

        if self.is_gpt_oss:
            content = parse_harmony_response(content)
        return content

    def get_usage(self) -> dict[str, Any]:
        """Return token usage stats and estimated cost."""
        with self._usage_lock:
            calls = self.total_calls
        return {
            "model": self.model,
            "calls": calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
        }

    def reset_usage(self):
        """Reset token counters."""
        with self._usage_lock:
            self.total_calls = 0
        self._provider.total_input_tokens = 0
        self._provider.total_output_tokens = 0
        self._provider.total_cost = 0.0

    def batch(self, prompts: list[str]) -> list[str]:
        """Parallel batch call. Returns results in same order as prompts."""
        if not prompts:
            return []

        prompts = [self._truncate(p) for p in prompts]
        results = self._provider.generate(
            prompts, system_prompt=self._system_prompt, **self._model_kwargs
        )
        contents = [r[0] for r in results]

        with self._usage_lock:
            self.total_calls += len(prompts)

        if self.is_gpt_oss:
            contents = [parse_harmony_response(c) for c in contents]
        return contents


def make_local_llm(
    model: str = "gpt-oss-120b",
    host: str = os.environ.get("LOCAL_LLM_HOST", "localhost"),
    port: int = int(os.environ.get("LOCAL_LLM_PORT", "30000")),
    max_tokens: int = 4096,
    max_workers: int = 16,
) -> LLM:
    """Create a local LLM caller against an OpenAI-compatible server.

    Usage:
        llm = make_local_llm()
        response = llm("Hello!")
    """
    return LLM(
        model=model,
        api_base=f"http://{host}:{port}/v1",
        max_tokens=max_tokens,
        max_workers=max_workers,
    )


def make_stub_llm(response: str = '{"reasoning": "stub", "final_answer": "stub"}'):
    """Create a stub LLM caller for testing."""
    return lambda prompt: response
