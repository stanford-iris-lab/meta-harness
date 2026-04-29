"""Evaluators for kept MCE and OOD tasks."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from .constants import TRANSFER_TASKS


def extract_final_answer(response_text: str) -> str:
    if not response_text:
        return ""

    text = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        payload = []
        in_json = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```json"):
                in_json = True
                continue
            if stripped.startswith("```") and in_json:
                break
            if in_json:
                payload.append(line)
        if payload:
            text = "\n".join(payload)
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:].strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "final_answer" in data:
            return str(data["final_answer"])
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "final_answer" in data:
                return str(data["final_answer"])
        except json.JSONDecodeError:
            pass

    return response_text


def eval_finer(prediction: str, target: str) -> bool:
    return extract_final_answer(prediction).lower().strip() == target.lower().strip()


def eval_uspto(prediction: str, target: str) -> dict:
    def parse_reactants(smiles: str) -> set[str]:
        return {
            part.strip().lower() for part in smiles.strip().split(".") if part.strip()
        }

    pred_set = parse_reactants(extract_final_answer(prediction))
    target_set = parse_reactants(target)
    if not pred_set and not target_set:
        jaccard = 1.0
    elif not pred_set or not target_set:
        jaccard = 0.0
    else:
        jaccard = len(pred_set & target_set) / len(pred_set | target_set)
    return {"correct": pred_set == target_set, "jaccard_similarity": jaccard}


def eval_symptom2disease(prediction: str, target: str) -> bool:
    text = extract_final_answer(prediction)
    match = re.search(r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", text, re.I | re.S)
    if match:
        text = match.group(1).strip()
    else:
        match = re.search(
            r"(?:diagnosis|final diagnosis|conclusion)[:：]\s*([^\n]+)", text, re.I
        )
        if match:
            text = match.group(1).strip()

    def normalize(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"\s+", " ", value)
        return re.sub(r"[.!?]+$", "", value)

    return normalize(text) == normalize(target)


def eval_lawbench(prediction: str, target: str) -> dict:
    def parse_charges(text: str) -> set[str]:
        text = extract_final_answer(text).strip()
        match = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", text)
        if match:
            text = match.group(1).strip()
        elif "罪名:" in text:
            text = text.split("罪名:")[-1]
        text = re.sub(r"<eoa>.*", "", text).strip()
        for sep in [";", "；", ",", "，", "、"]:
            if sep in text:
                return {part.strip() for part in text.split(sep) if part.strip()}
        return {text} if text else set()

    pred_charges = parse_charges(prediction)
    true_charges = parse_charges(target)
    tp = len(pred_charges & true_charges)
    fp = len(pred_charges - true_charges)
    fn = len(true_charges - pred_charges)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correct": pred_charges == true_charges,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def eval_aegis(prediction: str, target: str) -> bool:
    pred = extract_final_answer(prediction).lower()
    if "unsafe" in pred:
        pred = "unsafe"
    elif "safe" in pred:
        pred = "safe"
    else:
        pred = pred.strip()
    return pred == target.lower().strip()


def eval_classification(prediction: str, target: str) -> bool:
    def normalize(text: str) -> str:
        text = extract_final_answer(text).lower().strip()
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"[.!?]+$", "", text)

    return normalize(prediction) == normalize(target)


def get_evaluator(task: str) -> Callable:
    if task == "FiNER":
        return lambda pred, target, **kwargs: eval_finer(pred, target)
    if task == "USPTO":

        def _eval_uspto(pred: str, target: str, **kwargs) -> dict:
            raw = eval_uspto(pred, target)
            return {
                "was_correct": raw["correct"],
                "metrics": {"jaccard_similarity": raw["jaccard_similarity"]},
            }

        return _eval_uspto
    if task == "Symptom2Disease":
        return lambda pred, target, **kwargs: eval_symptom2disease(pred, target)
    if task == "LawBench":

        def _eval_lawbench(pred: str, target: str, **kwargs) -> dict:
            raw = eval_lawbench(pred, target)
            return {
                "was_correct": raw["correct"],
                "metrics": {
                    "f1": raw["f1"],
                    "precision": raw["precision"],
                    "recall": raw["recall"],
                    "tp": raw["tp"],
                    "fp": raw["fp"],
                    "fn": raw["fn"],
                },
            }

        return _eval_lawbench
    if task == "AEGIS":
        return lambda pred, target, **kwargs: eval_aegis(pred, target)
    if task in TRANSFER_TASKS:
        return lambda pred, target, **kwargs: eval_classification(pred, target)
    raise ValueError(f"Unknown task: {task}")
