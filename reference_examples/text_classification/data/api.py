"""Dataset entrypoints used by inner_loop.py."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable

from .constants import MCE_TASKS, TRANSFER_TASKS
from .evaluators import get_evaluator
from .loaders import load_mce_dataset, load_transfer_dataset


def _balanced_subsample(examples: list[dict], num: int, seed: int = 42) -> list[dict]:
    by_label = defaultdict(list)
    for example in examples:
        by_label[example["target"]].append(example)
    rng = random.Random(seed)
    for label in by_label:
        rng.shuffle(by_label[label])

    labels = sorted(by_label)
    base = num // len(labels)
    remainder = num % len(labels)
    result = []
    for index, label in enumerate(labels):
        take = base + (1 if index < remainder else 0)
        result.extend(by_label[label][:take])

    if len(result) < num:
        used = {id(example) for example in result}
        for label in labels:
            for example in by_label[label]:
                if id(example) in used:
                    continue
                result.append(example)
                used.add(id(example))
                if len(result) >= num:
                    break
            if len(result) >= num:
                break

    rng.shuffle(result)
    return result[:num]


def _with_context(examples: list[dict]) -> list[dict]:
    result = []
    for example in examples:
        text = example["input"]
        if example.get("context"):
            text = f"{example['context']}\n\n{text}"
        result.append({"input": text, "target": example["target"]})
    return result


def load_dataset_for_eval(
    task: str,
    num_examples: int | None = None,
    shuffle_seed: int | None = None,
) -> tuple[list[dict], Callable]:
    if task in MCE_TASKS:
        examples = _with_context(load_mce_dataset(task))
    elif task in TRANSFER_TASKS:
        examples = _with_context(load_transfer_dataset(task))
    else:
        raise ValueError(f"Unknown task: {task}")

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(examples)
    if num_examples is not None:
        examples = examples[:num_examples]
    return examples, get_evaluator(task)


def load_dataset_splits(
    task: str,
    num_train: int,
    num_test: int = 20,
    shuffle_seed: int = 42,
) -> tuple[list[dict], list[dict], Callable]:
    evaluator = get_evaluator(task)

    if task in MCE_TASKS:
        train_examples = _with_context(load_mce_dataset(task, split="train"))
        test_examples = _with_context(load_mce_dataset(task, split="test"))
        if num_train > len(train_examples) or num_test > len(test_examples):
            raise ValueError(f"Requested too many examples for {task}")
        random.Random(shuffle_seed).shuffle(train_examples)
        random.Random(shuffle_seed).shuffle(test_examples)
        return train_examples[:num_train], test_examples[:num_test], evaluator

    if task in TRANSFER_TASKS:
        train_examples = _with_context(load_transfer_dataset(task, split="train"))
        test_examples = _with_context(load_transfer_dataset(task, split="test"))
        if num_train > len(train_examples) or num_test > len(test_examples):
            raise ValueError(f"Requested too many examples for {task}")
        return (
            _balanced_subsample(train_examples, num_train, seed=shuffle_seed),
            _balanced_subsample(test_examples, num_test, seed=shuffle_seed),
            evaluator,
        )

    raise ValueError(f"Unknown task: {task}")


def load_dataset_splits_3way(
    task: str,
    num_train: int = 30,
    num_val: int = 35,
    num_test: int = 35,
    shuffle_seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], Callable]:
    evaluator = get_evaluator(task)

    if task in MCE_TASKS:
        train_examples = _with_context(load_mce_dataset(task, split="train"))
        val_examples = _with_context(load_mce_dataset(task, split="val"))
        test_examples = _with_context(load_mce_dataset(task, split="test"))
        if (
            num_train > len(train_examples)
            or num_val > len(val_examples)
            or num_test > len(test_examples)
        ):
            raise ValueError(f"Requested too many examples for {task}")
        random.Random(shuffle_seed).shuffle(train_examples)
        random.Random(shuffle_seed).shuffle(val_examples)
        random.Random(shuffle_seed).shuffle(test_examples)
        return (
            train_examples[:num_train],
            val_examples[:num_val],
            test_examples[:num_test],
            evaluator,
        )

    if task in TRANSFER_TASKS:
        train_examples = _with_context(load_transfer_dataset(task, split="train"))
        val_examples = _with_context(load_transfer_dataset(task, split="val"))
        test_examples = _with_context(load_transfer_dataset(task, split="test"))
        if (
            num_train > len(train_examples)
            or num_val > len(val_examples)
            or num_test > len(test_examples)
        ):
            raise ValueError(f"Requested too many examples for {task}")
        return (
            _balanced_subsample(train_examples, num_train, seed=shuffle_seed),
            _balanced_subsample(val_examples, num_val, seed=shuffle_seed),
            _balanced_subsample(test_examples, num_test, seed=shuffle_seed),
            evaluator,
        )

    raise ValueError(f"Unknown task: {task}")
