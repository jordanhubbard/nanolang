"""I execute import-path contracts through the rebuilt self-hosted driver."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SelfhostImportPaths(unittest.TestCase):
    def test_compiled_production_path_helpers(self):
        driver = (ROOT / "src_nano/nanoc_v06.nano").read_text()
        parser = (ROOT / "src_nano/parser.nano").read_text()
        parts = []
        for source, name in [(driver, "nlc_str_starts_with"), (driver, "nlc_str_index_of"),
                             (parser, "parser_decode_import_path"), (driver, "parse_import_path_from_line")]:
            body = source.split("\nfn " + name + "(", 1)[1].split("\nshadow " + name, 1)[0]
            parts.append("fn " + name + "(" + body)
        with tempfile.TemporaryDirectory(prefix="nanolang-path-helpers-") as directory:
            path = Path(directory)
            source = path / "helpers.nano"
            source.write_text("\n".join(parts) + '\nfn main() -> int {\n'
                              '(println (parser_decode_import_path "plain.nano"))\n'
                              '(println (parse_import_path_from_line "module \\"plain.nano\\" as probe"))\n'
                              'return 0 }\nshadow main { assert (== (main) 0) }\n')
            compiler = str(ROOT / "bin/nanoc_stage1")
            result = subprocess.run([compiler, str(source), "-o", str(path / "helpers"), "-k"],
                                    cwd=ROOT, env=dict(os.environ, TMPDIR=directory),
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(path / "helpers")], capture_output=True, text=True)
            self.assertEqual(result.stdout.splitlines(), ["plain.nano", "plain.nano"],
                             (path / "nanolang_temp.c").read_text())

    def check_path(self, filename, suffix="", invalid=False):
        compiler = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage1"))
        with tempfile.TemporaryDirectory(prefix="nanolang-selfhost-import-") as directory:
            path = Path(directory)
            module = path / filename
            module.write_text("module path_probe\npub fn answer() -> int { return 42 }\n"
                              "shadow answer { assert (== (answer) 42) }\n")
            spelling = json.dumps(str(module))
            if invalid:
                spelling = spelling[:-1] + '\\0ignored"'
            source = path / "main.nano"
            source.write_text(f"module {spelling} as probe{suffix}\n"
                              "fn main() -> int { assert (== (probe.answer) 42) return 0 }\n"
                              "shadow main { assert (== (main) 0) }\n")
            output = path / "program"
            result = subprocess.run([str(compiler), str(source), "-o", str(output)],
                                    cwd=ROOT, env=dict(os.environ, TMPDIR=directory),
                                    capture_output=True, text=True, timeout=120)
            if invalid:
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(output.exists())
                self.assertIn("NUL-containing import path", result.stdout + result.stderr)
            else:
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-6000:])
                subprocess.run([str(output)], check=True, timeout=10)

    def test_plain(self):
        self.check_path("path_probe.nano")

    def test_comment_quotes_are_not_path(self):
        self.check_path("path_probe.nano", ' # "not a path"')

    def test_escaped_filename(self):
        self.check_path('path_probe"back\\slash\nline.nano')

    def test_nul_rejected_before_truncated_lookup(self):
        self.check_path("path_probe.nano", invalid=True)


if __name__ == "__main__":
    unittest.main()
