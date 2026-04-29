from __future__ import annotations

import unittest

from text_classification.data import (
    ALL_TASKS,
    TRANSFER_TASKS,
    extract_final_answer,
    get_evaluator,
    load_dataset_for_eval,
    load_dataset_splits,
    load_dataset_splits_3way,
    load_mce_dataset,
)


class DataTests(unittest.TestCase):
    def test_transfer_tasks_match_paper_subset(self) -> None:
        self.assertEqual(
            TRANSFER_TASKS,
            [
                "AGNews",
                "GoEmotions",
                "Banking77",
                "FinancialPhraseBank",
                "SciCite",
                "TweetEval_hate",
                "Amazon5",
                "SciTail",
            ],
        )
        for task in ["YahooAnswers", "DBpedia14", "Newsgroups20", "MNLI", "RTE"]:
            with self.subTest(task=task):
                self.assertNotIn(task, TRANSFER_TASKS)

    def test_supported_tasks_match_release(self) -> None:
        for task in [
            "USPTO",
            "Symptom2Disease",
            "LawBench",
            "FiNER",
            "AEGIS",
            "AGNews",
            "GoEmotions",
            "Banking77",
            "FinancialPhraseBank",
            "SciCite",
            "TweetEval_hate",
            "Amazon5",
            "SciTail",
        ]:
            with self.subTest(task=task):
                self.assertIn(task, ALL_TASKS)

    def test_all_vendored_splits_load(self) -> None:
        expected = {
            "FiNER": {"train": 200, "val": 100, "test": 100},
            "USPTO": {"train": 50, "val": 30, "test": 100},
            "Symptom2Disease": {"train": 200, "val": 50, "test": 212},
            "LawBench": {"train": 200, "val": 50, "test": 100},
            "AEGIS": {"train": 400, "val": 128, "test": 140},
        }
        for task, splits in expected.items():
            for split, count in splits.items():
                with self.subTest(task=task, split=split):
                    examples = load_mce_dataset(task, split)
                    self.assertEqual(len(examples), count)
                    self.assertTrue(examples[0]["input"])
                    self.assertTrue(examples[0]["target"])

    def test_eval_loader_limit_and_shuffle(self) -> None:
        examples, evaluator = load_dataset_for_eval(
            "USPTO",
            num_examples=7,
            shuffle_seed=42,
        )
        self.assertEqual(len(examples), 7)
        self.assertTrue(callable(evaluator))

    def test_two_way_split_uses_requested_counts(self) -> None:
        train, test, _ = load_dataset_splits(
            "LawBench",
            num_train=11,
            num_test=13,
            shuffle_seed=42,
        )
        self.assertEqual((len(train), len(test)), (11, 13))

    def test_three_way_split_uses_requested_counts(self) -> None:
        for task, counts in {
            "FiNER": (13, 17, 19),
            "USPTO": (50, 30, 100),
            "Symptom2Disease": (17, 19, 23),
            "LawBench": (29, 31, 37),
            "AEGIS": (41, 43, 47),
        }.items():
            with self.subTest(task=task):
                train, val, test, _ = load_dataset_splits_3way(
                    task,
                    num_train=counts[0],
                    num_val=counts[1],
                    num_test=counts[2],
                    shuffle_seed=42,
                )
                self.assertEqual((len(train), len(val), len(test)), counts)

    def test_prompt_shapes_match_expected_task_templates(self) -> None:
        examples = {
            "FiNER": load_mce_dataset("FiNER", "train")[0]["input"],
            "USPTO": load_mce_dataset("USPTO", "train")[0]["input"],
            "Symptom2Disease": load_mce_dataset("Symptom2Disease", "train")[0]["input"],
            "LawBench": load_mce_dataset("LawBench", "train")[0]["input"],
            "AEGIS": load_mce_dataset("AEGIS", "train")[0]["input"],
        }
        self.assertIn("Task Context:", examples["FiNER"])
        self.assertIn('"final_answer"', examples["FiNER"])
        self.assertIn("Retrosynthesis Problem:", examples["USPTO"])
        self.assertIn('"final_answer"', examples["USPTO"])
        self.assertIn("## Patient Symptoms", examples["Symptom2Disease"])
        self.assertIn(
            "[DIAGNOSIS]diagnosis_name[/DIAGNOSIS]", examples["Symptom2Disease"]
        )
        self.assertIn("## 案件事实", examples["LawBench"])
        self.assertIn("[罪名]", examples["LawBench"])
        self.assertIn("safe or unsafe", examples["AEGIS"].lower())

    def test_lawbench_loader_strips_answer_prefix(self) -> None:
        example = load_mce_dataset("LawBench", "train")[0]
        self.assertNotIn("罪名:", example["target"])

    def test_extract_final_answer_handles_json_and_fallback(self) -> None:
        self.assertEqual(
            extract_final_answer('{"final_answer":"abc"}'),
            "abc",
        )
        self.assertEqual(
            extract_final_answer('```json\n{"final_answer":"xyz"}\n```'),
            "xyz",
        )
        self.assertEqual(extract_final_answer("raw text"), "raw text")

    def test_uspto_evaluator_reports_metrics(self) -> None:
        evaluator = get_evaluator("USPTO")
        result = evaluator(
            '{"final_answer":"A.B"}',
            "b.a",
        )
        self.assertTrue(result["was_correct"])
        self.assertEqual(result["metrics"]["jaccard_similarity"], 1.0)

    def test_symptom_evaluator_accepts_tagged_answer(self) -> None:
        evaluator = get_evaluator("Symptom2Disease")
        self.assertTrue(
            evaluator(
                "[DIAGNOSIS]Diabetes[/DIAGNOSIS]",
                "diabetes",
            )
        )

    def test_lawbench_evaluator_reports_metrics(self) -> None:
        evaluator = get_evaluator("LawBench")
        result = evaluator("[罪名]盗窃;诈骗<eoa>", "罪名:盗窃;诈骗")
        self.assertTrue(result["was_correct"])
        self.assertEqual(result["metrics"]["tp"], 2)

    def test_finer_evaluator_runs(self) -> None:
        evaluator = get_evaluator("FiNER")
        self.assertTrue(
            evaluator(
                '{"final_answer":"DebtInstrumentInterestRateStatedPercentage"}',
                "DebtInstrumentInterestRateStatedPercentage",
            )
        )

    def test_aegis_evaluator_runs(self) -> None:
        evaluator = get_evaluator("AEGIS")
        self.assertTrue(evaluator("unsafe", "unsafe"))


if __name__ == "__main__":
    unittest.main()
