"""
Minimal wrapper around `claude -p` for programmatic usage with logging.
Calls Claude Code CLI via subprocess, parses stream-json output,
tracks tool calls / file reads / token usage, and logs everything to disk.
Works independently of your local Claude Code setup (skills/plugins not inherited)
"""

import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_DIR = os.environ.get("CLAUDE_WRAPPER_LOG_DIR", "experience")
_EMPTY_PLUGIN_DIR = Path(__file__).parent / ".empty_plugins"

# Common tool sets
TOOLS_READ = ["Read", "Glob", "Grep"]
TOOLS_WRITE = ["Read", "Glob", "Grep", "Edit", "Write"]
TOOLS_BASH = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]
TOOLS_ALL = TOOLS_BASH + ["Agent", "WebSearch", "WebFetch"]


def _slugify(text, max_words=4):
    """Create a short slug from text for directory names."""
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return "-".join(words[:max_words]) or "run"


def _clean_read_output(output):
    """Strip line number prefixes (e.g. '     1→') from Read tool output."""
    lines = []
    for line in output.split("\n"):
        m = re.match(r"\s*\d+\u2192(.*)", line)
        lines.append(m.group(1) if m else line)
    return "\n".join(lines)


def _count_read_lines(output):
    """Count numbered lines in Read tool output."""
    return sum(1 for line in output.split("\n") if re.match(r"\s*\d+\u2192", line))


@dataclass
class ToolCall:
    name: str
    tool_id: str
    input: dict
    output: str = ""
    is_error: bool = False


@dataclass
class SessionResult:
    prompt: str
    text: str
    tool_calls: list
    files_read: dict  # {path: {"reads": N, "lines": M}}
    files_written: dict  # {path: {"lines_written": M}}
    token_usage: dict
    duration_seconds: float
    model: str
    session_id: str
    exit_code: int
    cost_usd: float
    raw_events: list
    command: list = None
    cwd: str = None
    stderr: str = ""
    skill: dict = None
    name: str = None
    log_dir: str = None

    def show(self):
        """Print compact one-line-per-event summary."""
        if self.exit_code != 0:
            print(f"  FAILED (exit={self.exit_code})")
            print(f"  {(self.stderr or 'No stderr.')[:300]}")
            return
        for tc in self.tool_calls:
            inp = tc.input
            arg = inp.get("file_path") or inp.get("pattern") or ""
            if not arg and "command" in inp:
                arg = inp["command"][:120]
            if not arg and "description" in inp:
                arg = inp["description"][:120]
            if not arg and "prompt" in inp:
                arg = inp["prompt"][:120]
            err = " ERR" if tc.is_error else ""
            print(f"  tool: {tc.name}({arg}){err}")
        text = self.text.strip().replace("\n", " ")
        if text:
            print(f"  text: {text[:200]}")
        if self.files_read:
            items = ", ".join(
                f"{p}({v['reads']}x, {v['lines']}L)" for p, v in self.files_read.items()
            )
            print(f"  read: {items}")
        if self.files_written:
            items = ", ".join(
                f"{p}({v['lines_written']}L)" for p, v in self.files_written.items()
            )
            print(f"  wrote: {items}")
        print(
            f"  {self.token_usage['input_tokens']}in/"
            f"{self.token_usage['output_tokens']}out  "
            f"${self.cost_usd:.4f}  {self.duration_seconds:.1f}s"
        )


def build_command(
    prompt,
    model="sonnet",
    allowed_tools=None,
    system_prompt=None,
    tools=None,
    disallowed_tools=None,
    disable_skills=True,
    disable_mcp=True,
    effort=None,
):
    """Build the claude CLI command list."""
    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--setting-sources",
        "",
    ]

    effective_tools = tools if tools is not None else allowed_tools
    if effective_tools:
        cmd.extend(["--tools", ",".join(effective_tools)])

    if allowed_tools:
        cmd.append("--allowedTools")
        cmd.extend(allowed_tools)

    if disallowed_tools:
        cmd.append("--disallowedTools")
        cmd.extend(disallowed_tools)

    if disable_skills:
        cmd.append("--disable-slash-commands")

    if disable_mcp:
        cmd.append("--strict-mcp-config")

    _EMPTY_PLUGIN_DIR.mkdir(exist_ok=True)
    cmd.extend(["--plugin-dir", str(_EMPTY_PLUGIN_DIR)])

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    if effort:
        cmd.extend(["--effort", effort])

    return cmd


def _make_relative(filepath, cwd):
    """Convert absolute path to relative if it's under cwd."""
    if not cwd or not filepath:
        return filepath
    try:
        return os.path.relpath(filepath, cwd)
    except ValueError:
        return filepath


def parse_stream_events(stdout, prompt, model, duration, exit_code, cwd=None):
    """Parse newline-delimited JSON from stream-json output."""
    events = []
    text_parts = []
    tool_calls = []
    tool_call_map = {}
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    session_id = ""
    cost_usd = 0.0

    for line in stdout.strip().split("\n") if stdout.strip() else []:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        events.append(event)
        etype = event.get("type", "")

        if etype == "assistant":
            msg = event.get("message", {})
            usage = msg.get("usage", {})
            token_usage["input_tokens"] += usage.get("input_tokens", 0)
            token_usage["output_tokens"] += usage.get("output_tokens", 0)
            for cache_key in (
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                if cache_key in usage:
                    token_usage[cache_key] = (
                        token_usage.get(cache_key, 0) + usage[cache_key]
                    )

            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")

                if btype == "text":
                    text_parts.append(block["text"])
                elif btype == "tool_use":
                    tc = ToolCall(
                        name=block["name"],
                        tool_id=block.get("id", ""),
                        input=block.get("input", {}),
                    )
                    tool_calls.append(tc)
                    tool_call_map[tc.tool_id] = tc

        elif etype == "user":
            msg = event.get("message", {})
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid in tool_call_map:
                        tool_call_map[tid].output = str(block.get("content", ""))
                        tool_call_map[tid].is_error = block.get("is_error", False)

        elif etype == "result":
            session_id = event.get("session_id", "")
            cost_usd = event.get("total_cost_usd", 0.0)
            result_usage = event.get("usage", {})
            if result_usage:
                token_usage["input_tokens"] = result_usage.get(
                    "input_tokens", token_usage["input_tokens"]
                )
                token_usage["output_tokens"] = result_usage.get(
                    "output_tokens", token_usage["output_tokens"]
                )

    # Compute file stats from completed tool calls
    files_read = {}
    files_written = {}
    for tc in tool_calls:
        if tc.name == "Read" and "file_path" in tc.input:
            path = _make_relative(tc.input["file_path"], cwd)
            lines = _count_read_lines(tc.output)
            if path in files_read:
                files_read[path]["reads"] += 1
                files_read[path]["lines"] += lines
            else:
                files_read[path] = {"reads": 1, "lines": lines}
        elif tc.name == "Write" and "file_path" in tc.input:
            path = _make_relative(tc.input["file_path"], cwd)
            content = tc.input.get("content", "")
            lines = content.count("\n") + (1 if content else 0)
            files_written[path] = {"lines_written": lines}
        elif tc.name == "Edit" and "file_path" in tc.input:
            path = _make_relative(tc.input["file_path"], cwd)
            new_str = tc.input.get("new_string", "")
            lines = new_str.count("\n") + (1 if new_str else 0)
            if path in files_written:
                files_written[path]["lines_written"] += lines
            else:
                files_written[path] = {"lines_written": lines}

    return SessionResult(
        prompt=prompt,
        text="".join(text_parts),
        tool_calls=tool_calls,
        files_read=files_read,
        files_written=files_written,
        token_usage=token_usage,
        duration_seconds=duration,
        model=model,
        session_id=session_id,
        exit_code=exit_code,
        cost_usd=cost_usd,
        raw_events=events,
    )


def _extract_json_blocks(text):
    """Extract named JSON code blocks from response text.

    Looks for patterns like:
        **`logs/pending_eval.json`:**
        ```json
        { ... }
        ```
    Returns list of (filename, parsed_json) tuples.
    """
    results = []
    # Match: optional bold/backtick filename hint, then ```json block
    pattern = re.compile(
        r"(?:\*\*`?([^`*\n]+\.json)`?\*\*[: \t]*\n)?"
        r"```json\s*\n(.*?)```",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        name_hint = m.group(1)
        body = m.group(2).strip()
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        filename = Path(name_hint).name if name_hint else None
        results.append((filename, parsed))
    return results


def log_session(result, log_dir):
    """Write session to a directory. Returns the directory path.

    Structure:
        <log_dir>/<timestamp>_<slug>/
            meta.json      - prompt, model, tokens, cost, file stats
            response.md    - text output
            events.jsonl   - raw stream events
            artifacts/     - JSON blocks extracted from response
            tools/
                001_Read.txt   - per-tool-call, human-readable
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = result.name or _slugify(result.prompt)
    run_dir = Path(log_dir) / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # meta.json - compact, scannable (no raw events or tool outputs)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": result.prompt,
        "model": result.model,
        "session_id": result.session_id,
        "exit_code": result.exit_code,
        "duration_seconds": round(result.duration_seconds, 2),
        "cost_usd": result.cost_usd,
        "token_usage": result.token_usage,
        "command": result.command,
        "cwd": result.cwd,
        "skill": result.skill,
        "files_read": result.files_read,
        "files_written": result.files_written,
        "tool_summary": [
            f"{tc.name}({'ERR ' if tc.is_error else ''}"
            f"{tc.input.get('file_path') or tc.input.get('pattern') or tc.input.get('command', '')[:120] or tc.input.get('description', '')[:120]})"
            for tc in result.tool_calls
        ],
    }
    if result.stderr:
        meta["stderr"] = result.stderr
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # response.md
    if result.text:
        (run_dir / "response.md").write_text(result.text)

    # artifacts/ - JSON blocks extracted from response text
    if result.text:
        json_blocks = _extract_json_blocks(result.text)
        if json_blocks:
            art_dir = run_dir / "artifacts"
            art_dir.mkdir(exist_ok=True)
            for i, (name, data) in enumerate(json_blocks, 1):
                fname = name or f"{i:03d}.json"
                (art_dir / fname).write_text(json.dumps(data, indent=2) + "\n")

    # events.jsonl
    if result.raw_events:
        lines = [json.dumps(e, default=str) for e in result.raw_events]
        (run_dir / "events.jsonl").write_text("\n".join(lines) + "\n")

    # tools/ - one human-readable file per tool call
    if result.tool_calls:
        tools_dir = run_dir / "tools"
        tools_dir.mkdir(exist_ok=True)
        for i, tc in enumerate(result.tool_calls, 1):
            parts = []

            # Header
            file_path = tc.input.get("file_path", "")
            if file_path:
                file_path = _make_relative(file_path, result.cwd)
            header = f"{tc.name}: {file_path}" if file_path else tc.name
            if tc.is_error:
                header += " [ERROR]"
            parts.append(header)
            parts.append("")

            # Input fields (skip file_path, already in header)
            for k, v in tc.input.items():
                if k == "file_path":
                    continue
                val = str(v)
                if "\n" in val or len(val) > 80:
                    parts.append(f"{k}:")
                    parts.append(val)
                    parts.append("")
                else:
                    parts.append(f"{k}: {v}")

            # Output (clean Read output of line-number prefixes)
            if tc.output:
                output = (
                    _clean_read_output(tc.output) if tc.name == "Read" else tc.output
                )
                parts.append("")
                parts.append("--- output ---")
                parts.append(output)

            (tools_dir / f"{i:03d}_{tc.name}.txt").write_text("\n".join(parts))

    result.log_dir = str(run_dir)
    return str(run_dir)


def load_skill(skill_path):
    """Load a skill markdown file. Returns content string or None if not found."""
    path = Path(skill_path)
    if path.exists():
        return path.read_text()
    return None


def load_skills(skills, skill_dir=None):
    """Load one or more skills by path, name, or from a directory."""
    if skill_dir is None:
        skill_dir = ".claude/skills"
    skill_dir = Path(skill_dir)
    loaded = []

    for s in skills:
        p = Path(s)
        if p.is_dir() and (p / "SKILL.md").is_file():
            skill_file = p / "SKILL.md"
            loaded.append(
                {
                    "path": str(skill_file),
                    "name": p.name,
                    "content": skill_file.read_text(),
                }
            )
        elif p.is_dir():
            for md in sorted(p.glob("*.md")):
                loaded.append(
                    {"path": str(md), "name": md.stem, "content": md.read_text()}
                )
        elif p.is_file():
            loaded.append({"path": str(p), "name": p.stem, "content": p.read_text()})
        else:
            candidates = [
                skill_dir / s / "SKILL.md",
                skill_dir / s,
                skill_dir / f"{s}.md",
            ]
            for c in candidates:
                if c.is_file():
                    name = c.parent.name if c.name == "SKILL.md" else c.stem
                    loaded.append(
                        {"path": str(c), "name": name, "content": c.read_text()}
                    )
                    break

    return loaded


def _default_progress(event, tool_calls):
    """Default progress callback: print one line per tool call to stderr."""
    if event.get("type") != "assistant":
        return
    for block in event.get("message", {}).get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            inp = block.get("input", {})
            arg = inp.get("file_path") or inp.get("pattern") or ""
            if not arg and "command" in inp:
                arg = inp["command"][:120]
            if not arg and "description" in inp:
                arg = inp["description"][:120]
            if not arg and "prompt" in inp:
                arg = inp["prompt"][:120]
            arg = arg.replace("\n", " ").strip()
            n = len(tool_calls)
            print(f"  [{n}] {block['name']}({arg[:120]})", flush=True)


def _enqueue_lines(pipe, q, stream_name):
    """Read lines from a pipe in a background thread and push them into a queue."""
    try:
        for line in iter(pipe.readline, ""):
            q.put((stream_name, line))
    finally:
        pipe.close()


def run(
    prompt,
    model="sonnet",
    allowed_tools=None,
    tools=None,
    disallowed_tools=None,
    cwd=None,
    log_dir=None,
    name=None,
    system_prompt=None,
    skill_path=None,
    skills=None,
    skill_dir=None,
    timeout_seconds=None,
    disable_skills=True,
    disable_mcp=True,
    progress=True,
    effort=None,
):
    """Run `claude -p` and return parsed SessionResult. Logs to log_dir.

    Uses your Claude Pro/Max subscription (not API key).

    Args:
        progress: Show live progress. True = default printer, callable = custom
                  callback(event, tool_calls_so_far), False/None = silent.
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    if allowed_tools is None:
        allowed_tools = list(TOOLS_BASH)
    if disallowed_tools is None:
        disallowed_tools = []

    # Load skills
    all_skills = []
    if skill_path:
        content = load_skill(skill_path)
        if content:
            all_skills.append(
                {"path": skill_path, "name": Path(skill_path).stem, "content": content}
            )
    if skills:
        all_skills.extend(load_skills(skills, skill_dir))

    # Inject skill content into system prompt
    skill_info = all_skills if all_skills else None
    if all_skills:
        skill_text = "\n\n".join(
            f"## Skill: {s['name']}\n{s['content']}" for s in all_skills
        )
        prefix = f"Follow these skill instructions:\n\n{skill_text}\n\n"
        system_prompt = prefix + (system_prompt or "")

    cmd = build_command(
        prompt,
        model,
        allowed_tools,
        system_prompt,
        tools=tools,
        disallowed_tools=disallowed_tools,
        disable_skills=disable_skills,
        disable_mcp=disable_mcp,
        effort=effort,
    )

    effective_cwd = cwd or os.getcwd()

    env = os.environ.copy()
    # Use API key if available (cheaper, no subscription needed), else subscription auth
    if "ANTHROPIC_API_KEY" not in env:
        pass  # will use subscription auth automatically

    # Resolve progress callback
    if progress is True:
        on_event = _default_progress
    elif callable(progress):
        on_event = progress
    else:
        on_event = None

    start = time.time()
    stdout_lines = []
    stderr_lines = []
    exit_code = 0
    # Track tool calls during streaming for progress callback
    _live_tool_calls = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
            env=env,
        )
        deadline = start + timeout_seconds if timeout_seconds else None
        q = queue.Queue()
        stdout_thread = threading.Thread(
            target=_enqueue_lines,
            args=(proc.stdout, q, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_enqueue_lines,
            args=(proc.stderr, q, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        while True:
            if deadline and time.time() > deadline:
                proc.kill()
                stderr_lines.append(
                    f"\nProcess timed out after {timeout_seconds} seconds."
                )
                exit_code = 124
                break

            try:
                stream_name, line = q.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue

            if stream_name == "stdout":
                stdout_lines.append(line)
                if on_event:
                    try:
                        event = json.loads(line)
                        # Track tool calls for progress counter
                        if event.get("type") == "assistant":
                            for block in event.get("message", {}).get("content", []):
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "tool_use"
                                ):
                                    _live_tool_calls.append(block)
                        on_event(event, _live_tool_calls)
                    except (json.JSONDecodeError, ValueError):
                        pass
            else:
                stderr_lines.append(line)

        proc.wait()
        if exit_code == 0:
            exit_code = proc.returncode
    except FileNotFoundError as e:
        stderr_lines = [str(e)]
        exit_code = 127
    duration = time.time() - start

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    result = parse_stream_events(
        stdout, prompt, model, duration, exit_code, cwd=effective_cwd
    )
    result.command = cmd
    result.cwd = effective_cwd
    result.stderr = stderr
    result.skill = skill_info
    result.name = name
    log_session(result, log_dir)
    return result


if __name__ == "__main__":
    import hashlib as _hashlib

    LOG_DIR = "experience"

    print("=== Test 1: Summarize this repo ===")
    run(
        "Read through all important files in this directory and give a summary. Only return the summary, no other text.",
        allowed_tools=TOOLS_READ,
        name="summarize-repo",
        log_dir=LOG_DIR,
    ).show()
    print()

    print("=== Test 2: Write 5 files ===")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    h = _hashlib.md5(ts.encode()).hexdigest()[:8]
    work_dir = f"/tmp/{ts}_{h}"
    os.makedirs(work_dir, exist_ok=True)
    run(
        "Create 5 Python files named 001.py through 005.py in /tmp/meta-harness-test. "
        "If the directory already exists, delete it and create a new one. "
        "Each should contain a single python function named task_N (where N is the file number) "
        "that returns N in a creative way. Nothing else.",
        allowed_tools=TOOLS_BASH,
        cwd=work_dir,
        name="write-5-files",
        log_dir=LOG_DIR,
    ).show()
    print(f"  dir: {work_dir}")
    print()

    print("=== Test 3: Web search ===")
    r = run(
        "Search the web for the 'Meta-Harness' paper and give a summary of the core idea. Brainstorm a list of 3 new _very specific_ applications of Meta-Harness, and tell me which you think is the most well-scoped and interesting. For that idea, point me to a small set of the best resources online for getting started",
        allowed_tools=TOOLS_ALL,
        name="websearch-test",
        log_dir=LOG_DIR,
    )
    r.show()
    print()
    print(r.text)
    print()

    print(f"Logs: {LOG_DIR}/")
