"""I check explicit opcode coverage, not the semantics of each transfer."""

from pathlib import Path
import re
import unittest


class OpcodeCoverage(unittest.TestCase):
    def test_classifier_and_emitter_handle_the_same_opcodes(self):
        source = (Path(__file__).resolve().parents[1] / "src/nanoisa/nvm2c.c").read_text()
        source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)

        def cases(signature):
            body = source.split(signature, 1)[1].split("\nstatic ", 1)[0]
            result = set(re.findall(r"\bcase\s+(OP_\w+)\s*:", body))
            self.assertIn("OP_JMP", result)
            self.assertIn("OP_CAST_INT", result)
            return result

        self.assertEqual(cases("static int classify_function_body("),
                         cases("static void emit_function_body("))


if __name__ == "__main__":
    unittest.main()
