"""Regression checks for lexical code-generation environments."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRANSPILER = ROOT / "src_nano" / "transpiler.nano"


class GenEnvScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TRANSPILER.read_text()

    def test_environment_extension_copies_all_parallel_arrays(self):
        body = self.source.split("fn genenv_copy", 1)[1].split("shadow genenv_copy", 1)[0]
        for field in ("names", "types", "mut_flags", "global_flags"):
            self.assertIn(f"env.{field}", body)
        self.assertIn("let copy: GenEnv = (genenv_copy env)", self.source)

    def test_alias_shadow_regression_is_exercised(self):
        shadow = self.source.split("shadow genenv_copy", 1)[1].split("fn genenv_put", 1)[0]
        self.assertIn('"selected" "fn() -> int"', shadow)
        self.assertIn('"selected" "int"', shadow)
        self.assertIn("(array_length outer.names) 1", shadow)

    def test_match_and_loop_binders_extend_a_scoped_environment(self):
        self.assertIn("let env_arm: GenEnv = (genenv_put env binding", self.source)
        self.assertGreaterEqual(self.source.count("let body_env: GenEnv = (genenv_put env for_stmt.var_name"), 3)


if __name__ == "__main__":
    unittest.main()
