"""I read binary streams without requiring a seekable source."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''
fn main() -> int {
    let bytes: array<u8> = (file_read_bytes "/dev/stdin")
    assert (== (array_length bytes) 8193)
    for i in (range 0 8193) { assert (== (at bytes i) (% i 256)) }
    return 0
}
shadow main { assert (== (main) 0) }
'''


class FileBytes(unittest.TestCase):
    def test_pipe_in_shadows_and_artifact(self):
        payload = bytes(i % 256 for i in range(8193))
        for compiler, vm in (("nanoc_c", False), ("nano_virt", True)):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-byte-pipe-") as tmp:
                source = Path(tmp) / "pipe.nano"
                source.write_text(SOURCE)
                output = Path(tmp) / ("pipe.nvm" if vm else "pipe")
                command = [str(ROOT / "bin" / compiler), str(source), "-o", str(output)]
                if vm:
                    command.append("--emit-nvm")
                built = subprocess.run(command, cwd=ROOT, input=payload, capture_output=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace"))
                command = [str(ROOT / "bin/nano_vm"), str(output)] if vm else [str(output)]
                run = subprocess.run(command, input=payload, capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
                self.assertEqual(run.stdout, b"")


if __name__ == "__main__":
    unittest.main()
