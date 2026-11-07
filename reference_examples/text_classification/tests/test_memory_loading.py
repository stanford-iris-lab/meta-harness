from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from text_classification import agents
from text_classification.agents.fewshot_all import FewShotAll
from text_classification.agents.no_memory import NoMemory
from text_classification.inner_loop import load_memory_system


class MemoryLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        agents.__path__.append(directory.name)
        self.addCleanup(agents.__path__.remove, directory.name)

    def write_candidate(self, name: str, source: str) -> str:
        (self.root / f"{name}.py").write_text(source)
        self.addCleanup(sys.modules.pop, f"text_classification.agents.{name}", None)
        return name

    def test_loads_local_subclass_regardless_of_name_order(self) -> None:
        for class_name in ("AFewShotMemory", "ZFewShotMemory"):
            with self.subTest(class_name=class_name):
                name = self.write_candidate(
                    f"test_{class_name.lower()}",
                    "from .fewshot_memory import FewShotMemory\n"
                    f"class {class_name}(FewShotMemory):\n"
                    "    def __init__(self, llm):\n"
                    "        super().__init__(llm, max_examples=9999)\n",
                )
                for path in (name, f"agents/{name}.py"):
                    with self.subTest(path=path):
                        memory = load_memory_system(path, llm=lambda _: "")
                        self.assertEqual(type(memory).__name__, class_name)
                        self.assertEqual(memory.max_examples, 9999)

    def test_imported_base_alone_is_not_a_candidate(self) -> None:
        name = self.write_candidate(
            "test_import_only", "from .fewshot_memory import FewShotMemory\n"
        )

        with self.assertRaisesRegex(ValueError, "No MemorySystem subclass"):
            load_memory_system(f"agents/{name}.py", llm=lambda _: "")

    def test_module_without_memory_class_is_rejected(self) -> None:
        name = self.write_candidate("test_no_memory_class", "class Unrelated: pass\n")

        with self.assertRaisesRegex(ValueError, "not found"):
            load_memory_system(name, llm=lambda _: "")

    def test_shipped_baselines_still_load(self) -> None:
        for name, expected_class in (
            ("no_memory", NoMemory),
            ("fewshot_all", FewShotAll),
        ):
            with self.subTest(name=name):
                memory = load_memory_system(name, llm=lambda _: "")
                self.assertIs(type(memory), expected_class)


if __name__ == "__main__":
    unittest.main()
