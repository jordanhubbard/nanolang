"""I keep text-file stream and embedded-NUL behavior consistent."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FileText(unittest.TestCase):
    def test_pipe_content_and_rejection(self):
        text = "aé\n" * 2049 + "end"
        cases = ((text.encode(), text), (b"", ""), (b"prefix\0suffix", ""))
        for compiler, vm, imported in (("nanoc_c", False, False), ("nano_virt", True, False),
                                      ("nanoc_c", False, True), ("nano_virt", True, True)):
            for index, (payload, expected) in enumerate(cases):
                with self.subTest(compiler=compiler, imported=imported, case=index), tempfile.TemporaryDirectory(prefix="nano-text-pipe-") as tmp:
                    source = Path(tmp) / "text.nano"
                    declaration = (f'module {json.dumps(str(ROOT / "modules/std/fs.nano"))} as fs\n'
                                   if imported else 'extern fn file_read(path: string) -> string\n')
                    reader = "fs.read" if imported else "file_read"
                    source.write_text(
                        declaration +
                        'fn main() -> int {\n'
                        ' unsafe {\n'
                        f' let actual: string = ({reader} "/dev/stdin")\n'
                        f' let expected: string = {json.dumps(expected, ensure_ascii=False)}\n'
                        ' assert (== actual expected)\n'
                        ' return 0\n}\n}\n'
                        'shadow main { assert (== (main) 0) }\n')
                    output = Path(tmp) / ("text.nvm" if vm else "text")
                    command = [str(ROOT / "bin" / compiler), str(source), "-o", str(output)]
                    if vm:
                        command.append("--emit-nvm")
                    built = subprocess.run(command, cwd=ROOT, input=payload, capture_output=True, timeout=60)
                    self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace")[:2000])
                    command = [str(ROOT / "bin/nano_vm"), str(output)] if vm else [str(output)]
                    run = subprocess.run(command, input=payload, capture_output=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace")[:2000])
                    self.assertEqual(run.stdout, b"")


if __name__ == "__main__":
    unittest.main()
