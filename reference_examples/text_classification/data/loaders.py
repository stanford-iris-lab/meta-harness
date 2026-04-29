"""Loaders for kept MCE and OOD tasks."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from .constants import (
    AG_NEWS_LABELS,
    AMAZON5_LABELS,
    FINANCIAL_PHRASEBANK_LABELS,
    FINER_CONTEXT,
    GOEMOTIONS_LABELS,
    MCE_DATA_PATH,
    MCE_TASKS,
    SCICITE_LABELS,
    SCITAIL_LABELS,
    TRANSFER_TASKS,
    TWEETEVAL_HATE_LABELS,
)


def _load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples


def _get_mce_split_path(task_dir: str, split: str) -> str:
    if split in ["val", "validation", "valid"]:
        val_path = Path(MCE_DATA_PATH) / task_dir / "val.jsonl"
        valid_path = Path(MCE_DATA_PATH) / task_dir / "valid.jsonl"
        return str(val_path if val_path.exists() else valid_path)
    return str(Path(MCE_DATA_PATH) / task_dir / f"{split}.jsonl")


def load_mce_dataset(
    task: str, split: str = "test", limit: int | None = None
) -> list[dict]:
    if task not in MCE_TASKS:
        raise ValueError(f"Unknown MCE task: {task}. Must be one of {MCE_TASKS}")
    loaders = {
        "FiNER": _load_finer,
        "USPTO": _load_uspto,
        "Symptom2Disease": _load_symptom2disease,
        "LawBench": _load_lawbench,
        "AEGIS": _load_aegis,
    }
    return loaders[task](split, limit)


def _load_finer(split: str = "test", limit: int | None = None) -> list[dict]:
    data = _load_jsonl(_get_mce_split_path("finer", split), limit)
    examples = []
    for example in data:
        prompt = f"""You are an expert domain problem solver.

Task Context:
{FINER_CONTEXT}

Instructional Context:


Question: {example["question"]}

You MUST respond with a valid JSON object containing exactly two fields:
1. "reasoning": Your step-by-step analysis (string)
2. "final_answer": Your concise final answer (string)
"""
        examples.append(
            {
                "input": prompt,
                "target": example["target"],
                "raw_question": example["question"],
                "context": "",
            }
        )
    return examples


def _load_uspto(split: str = "test", limit: int | None = None) -> list[dict]:
    data = _load_jsonl(_get_mce_split_path("uspto", split), limit)
    examples = []
    for example in data:
        prompt = f"""You are an expert organic chemist specializing in retrosynthesis analysis.

Retrosynthesis Problem:
{example["question"]}

Strategic Context:


Instructions:
- Analyze the product structure and identify key functional groups and bonds
- Consider the reaction type and typical disconnection strategies
- Think through the retrosynthetic analysis step-by-step
- Propose the most likely precursor reactants based on the reaction mechanism
- Output SMILES strings separated by periods (.) for multiple reactants
- Ignore atom mapping numbers in your analysis

You MUST respond with a valid JSON object containing exactly two fields:
1. "reasoning": Your detailed step-by-step retrosynthetic analysis, including:
   - Product structure analysis (key functional groups, stereochemistry, etc.)
   - Reaction type identification and typical mechanisms
   - Disconnection strategy and bond-breaking analysis
   - Proposed precursor structures and why they make sense
   - Verification that the forward reaction would yield the product
2. "final_answer": The SMILES string(s) of precursor reactants ONLY, separated by periods if multiple reactants (e.g., "CC(=O)Cl.c1ccccc1O")

Example response format:
{{
  "reasoning": "Your step-by-step retrosynthetic analysis... (less than 200 words)",
  "final_answer": "O=C=O.c1ccc(CO)cc1.C1CNCC1O"
}}"""
        examples.append(
            {
                "input": prompt,
                "target": example["target"],
                "raw_question": example["question"],
                "context": "",
            }
        )
    return examples


def _load_symptom2disease(split: str = "test", limit: int | None = None) -> list[dict]:
    data = _load_jsonl(_get_mce_split_path("symptom_diagnosis", split), limit)
    examples = []
    for example in data:
        prompt = f"""You are an expert medical diagnostician. Based on the patient's symptoms, provide a diagnosis.

Possible diagnoses include: drug reaction, allergy, chicken pox, diabetes, psoriasis, hypertension, cervical spondylosis, bronchial asthma, varicose veins, malaria, dengue, arthritis, impetigo, fungal infection, common cold, gastroesophageal reflux disease, urinary tract infection, typhoid, pneumonia, peptic ulcer disease, jaundice, migraine.

Please analyze the symptoms step by step, then provide your final diagnosis in the format:
[DIAGNOSIS]diagnosis_name[/DIAGNOSIS]

For example:
[DIAGNOSIS]diabetes[/DIAGNOSIS]


## Patient Symptoms
{example["question"]}

Please provide your reasoning and final diagnosis."""
        examples.append(
            {
                "input": prompt,
                "target": example["answer"],
                "raw_question": example["question"],
                "context": "",
            }
        )
    return examples


def _load_lawbench(split: str = "test", limit: int | None = None) -> list[dict]:
    data = _load_jsonl(_get_mce_split_path("crime_prediction", split), limit)
    examples = []
    for example in data:
        prompt = f"""请你模拟法官依据下面事实给出罪名。请先进行推理分析，然后将最终答案写在[罪名]和<eoa>之间。

格式示例：
- 单个罪名: [罪名]盗窃<eoa>
- 多个罪名: [罪名]盗窃;诈骗<eoa>


## 案件事实
{example["question"]}"""
        answer = example["answer"]
        if "罪名:" in answer:
            answer = answer.split("罪名:")[-1].strip()
        examples.append(
            {
                "input": prompt,
                "target": answer,
                "raw_question": example["question"],
                "context": "",
            }
        )
    return examples


def _load_aegis(split: str = "test", limit: int | None = None) -> list[dict]:
    data = _load_jsonl(_get_mce_split_path("aegis2", split), limit)
    examples = []
    for example in data:
        prompt = f"""
Task: Check if there is unsafe content in the user messages in the below conversation in accordance with our safety policy with the below INSTRUCTIONAL CONTEXT. Provide a safe or unsafe 'rating' for both the user message. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.
<BEGIN INSTRUCTIONAL CONTEXT>

<END INSTRUCTIONAL CONTEXT>

<BEGIN CONVERSATION>
user:
{example["prompt"]}
<END CONVERSATION>

You MUST respond with a valid JSON object containing exactly three fields:
1. "reasoning": Your reasoning for the answer (string)
2. "Safety_Categories": String of violated safety categories. Empty string "" if safe. (string)
3. "final_answer": Your safety classification - MUST be either "safe" or "unsafe" (string)
"""
        examples.append(
            {
                "input": prompt,
                "target": example["prompt_label"],
                "raw_question": example["prompt"],
                "context": "",
            }
        )
    return examples


def _hf_train_val_split(data: list, split: str) -> list:
    boundary = int(len(data) * 0.8)
    return data[boundary:] if split in ("val", "validation") else data[:boundary]


def _build_examples(data, text_fn, label_fn, prompt_fn, limit=None):
    examples = []
    for example in data:
        text = text_fn(example)
        label = label_fn(example)
        if label is None:
            continue
        examples.append(
            {
                "input": prompt_fn(text),
                "target": label,
                "raw_question": text,
                "context": "",
            }
        )
        if limit and len(examples) >= limit:
            break
    return examples


_JSON_OUTPUT_FMT = """You MUST respond with a valid JSON object containing exactly two fields:
1. "reasoning": Your step-by-step analysis (string)
2. "final_answer": {answer_desc}"""


def load_transfer_dataset(
    task: str, split: str = "test", limit: int | None = None
) -> list[dict]:
    if task not in TRANSFER_TASKS:
        raise ValueError(
            f"Unknown transfer task: {task}. Must be one of {TRANSFER_TASKS}"
        )
    loaders = {
        "AGNews": _load_ag_news,
        "GoEmotions": _load_goemotions,
        "Banking77": _load_banking77,
        "FinancialPhraseBank": _load_financial_phrasebank,
        "SciCite": _load_scicite,
        "TweetEval_hate": _load_tweeteval_hate,
        "Amazon5": _load_amazon5,
        "SciTail": _load_scitail,
    }
    return loaders[task](split, limit)


def _load_ag_news(split: str = "test", limit: int | None = None) -> list[dict]:
    ds = load_dataset("fancyzhx/ag_news")
    data = (
        list(ds["test"])
        if split == "test"
        else _hf_train_val_split(list(ds["train"]), split)
    )
    labels_str = ", ".join(AG_NEWS_LABELS)
    output_fmt = _JSON_OUTPUT_FMT.format(answer_desc="The category name (string)")

    def prompt_fn(text):
        return f"""You are a news topic classifier. Classify the article into exactly one category.

Categories: {labels_str}

Article:
{text}

{output_fmt}"""

    return _build_examples(
        data,
        text_fn=lambda ex: ex["text"],
        label_fn=lambda ex: AG_NEWS_LABELS[ex["label"]],
        prompt_fn=prompt_fn,
        limit=limit,
    )


def _load_goemotions(split: str = "test", limit: int | None = None) -> list[dict]:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    hf_split = "validation" if split in ("val", "validation") else split
    data = [example for example in ds[hf_split] if len(example["labels"]) == 1]
    labels_str = ", ".join(GOEMOTIONS_LABELS)
    output_fmt = _JSON_OUTPUT_FMT.format(
        answer_desc="The emotion name exactly as listed above (string)"
    )

    def prompt_fn(text):
        return f"""You are an emotion classifier. Classify the emotion expressed in the following text into exactly one category.

Emotions: {labels_str}

Text:
{text}

{output_fmt}"""

    return _build_examples(
        data,
        text_fn=lambda ex: ex["text"],
        label_fn=lambda ex: GOEMOTIONS_LABELS[ex["labels"][0]],
        prompt_fn=prompt_fn,
        limit=limit,
    )


def _load_banking77(split: str = "test", limit: int | None = None) -> list[dict]:
    ds = load_dataset("banking77")
    data = (
        list(ds["test"])
        if split == "test"
        else _hf_train_val_split(list(ds["train"]), split)
    )
    label_names = ds["train"].features["label"].names
    labels_str = ", ".join(label_names)
    output_fmt = _JSON_OUTPUT_FMT.format(
        answer_desc="The intent name exactly as listed above (string)"
    )

    def prompt_fn(text):
        return f"""You are a customer service intent classifier for a digital bank. Classify the customer query into exactly one intent.

Possible intents: {labels_str}

Customer query:
{text}

{output_fmt}"""

    return _build_examples(
        data,
        text_fn=lambda ex: ex["text"],
        label_fn=lambda ex: label_names[ex["label"]],
        prompt_fn=prompt_fn,
        limit=limit,
    )


def _load_financial_phrasebank(
    split: str = "test", limit: int | None = None
) -> list[dict]:
    ds = load_dataset("FinanceMTEB/financial_phrasebank")
    if split == "test":
        data = list(ds["test"])
    elif split in ("val", "validation"):
        all_train = list(ds["train"])
        boundary = int(len(all_train) * 0.8)
        data = all_train[boundary:]
    else:
        all_train = list(ds["train"])
        boundary = int(len(all_train) * 0.8)
        data = all_train[:boundary]
    labels_str = ", ".join(FINANCIAL_PHRASEBANK_LABELS)
    output_fmt = _JSON_OUTPUT_FMT.format(
        answer_desc="One of 'negative', 'neutral', or 'positive' (string)"
    )

    def prompt_fn(text):
        return f"""Classify the financial sentiment of the following statement. Note: sentiment is from an investor's perspective — 'positive' means good for the stock/company, 'negative' means bad.

Categories: {labels_str}

Statement:
{text}

{output_fmt}"""

    return _build_examples(
        data,
        text_fn=lambda ex: ex["text"],
        label_fn=lambda ex: ex["label_text"],
        prompt_fn=prompt_fn,
        limit=limit,
    )


def _load_scicite(split: str = "test", limit: int | None = None) -> list[dict]:
    ds = load_dataset("allenai/scicite", revision="refs/convert/parquet")
    hf_split = "validation" if split in ("val", "validation") else split
    data = list(ds[hf_split])
    label_map = {0: "background", 1: "method", 2: "result"}
    labels_str = ", ".join(SCICITE_LABELS)
    output_fmt = _JSON_OUTPUT_FMT.format(
        answer_desc="One of 'background', 'method', or 'result' (string)"
    )

    def prompt_fn(text):
        return f"""Classify the intent of the following scientific citation. A citation can serve as:
- background: provides context or prior work
- method: the cited work's method/tool/data is used
- result: compares or contrasts results with the cited work

Categories: {labels_str}

Citation context:
{text}

{output_fmt}"""

    return _build_examples(
        data,
        text_fn=lambda ex: ex["string"],
        label_fn=lambda ex: label_map.get(ex["label"]),
        prompt_fn=prompt_fn,
        limit=limit,
    )


def _load_tweeteval_hate(split: str = "test", limit: int | None = None) -> list[dict]:
    ds = load_dataset("tweet_eval", "hate")
    hf_split = "validation" if split in ("val", "validation") else split
    data = list(ds[hf_split])
    labels_str = ", ".join(TWEETEVAL_HATE_LABELS)
    output_fmt = _JSON_OUTPUT_FMT.format(
        answer_desc="Either 'non_hate' or 'hate' (string)"
    )

    def prompt_fn(text):
        return f"""Classify whether the following tweet contains hate speech or not.

Categories: {labels_str}

Tweet:
{text}

{output_fmt}"""

    return _build_examples(
        data,
        text_fn=lambda ex: ex["text"],
        label_fn=lambda ex: TWEETEVAL_HATE_LABELS[ex["label"]],
        prompt_fn=prompt_fn,
        limit=limit,
    )


def _load_amazon5(split: str = "test", limit: int | None = None) -> list[dict]:
    ds = load_dataset("SetFit/amazon_reviews_multi_en")
    hf_split = "validation" if split in ("val", "validation") else split
    data = list(ds[hf_split])
    labels_str = ", ".join(AMAZON5_LABELS)
    output_fmt = _JSON_OUTPUT_FMT.format(
        answer_desc="One of '1 star', '2 stars', '3 stars', '4 stars', or '5 stars' (string)"
    )

    def prompt_fn(text):
        return f"""Predict the star rating (1-5) of the following product review.

Ratings: {labels_str}

Review:
{text}

{output_fmt}"""

    return _build_examples(
        data,
        text_fn=lambda ex: ex["text"],
        label_fn=lambda ex: AMAZON5_LABELS[ex["label"]],
        prompt_fn=prompt_fn,
        limit=limit,
    )


def _load_scitail(split: str = "test", limit: int | None = None) -> list[dict]:
    ds = load_dataset("allenai/scitail", "snli_format")
    hf_split = "validation" if split in ("val", "validation") else split
    data = list(ds[hf_split])
    labels_str = ", ".join(SCITAIL_LABELS)
    output_fmt = _JSON_OUTPUT_FMT.format(
        answer_desc="One of 'entailment' or 'neutral' (string)"
    )

    def prompt_fn(text):
        return f"""Determine whether the premise entails the hypothesis in a scientific context.

Categories: {labels_str}

{text}

{output_fmt}"""

    return _build_examples(
        data,
        text_fn=lambda ex: f"Premise: {ex['sentence1']}\nHypothesis: {ex['sentence2']}",
        label_fn=lambda ex: ex["gold_label"],
        prompt_fn=prompt_fn,
        limit=limit,
    )
