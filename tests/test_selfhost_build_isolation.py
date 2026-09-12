"""I keep each self-hosted compilation's source and native staging private."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve()


class SelfhostBuildIsolation(unittest.TestCase):
    def source(self, directory, index):
        directory.mkdir()
        helper = directory / "helper.nano"
        helper.write_text(f"pub fn value() -> int {{ return {index + 11} }}\n" + "# padding\n" * 1000)
        source = directory / "input.nano"
        source.write_text('module "helper.nano" as helper\n'
                          'fn main() -> int { (println (int_to_string (helper.value))) return 0 }\n'
                          f'shadow main {{ assert (== (helper.value) {index + 11}) }}\n')
        return source

    def start(self, source, output, scratch, *options, cc=None):
        environment = dict(os.environ, TMPDIR=str(scratch))
        if cc is not None:
            environment["NANO_CC"] = cc
        return subprocess.Popen([str(COMPILER), str(source), "-o", str(output), *options],
                                cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)

    def finish(self, process):
        try:
            stdout, stderr = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return process.returncode, stdout + stderr

    def test_overlapping_source_and_native_compilations(self):
        with tempfile.TemporaryDirectory(prefix="nano-isolation-") as tmp:
            directory = Path(tmp)
            scratch = directory / "shared scratch's"
            scratch.mkdir()
            for target in ("c", "native"):
                with self.subTest(target=target):
                    jobs = []
                    try:
                        for index in range(4):
                            source = self.source(directory / f"{target}-{index}", index)
                            output = source.parent / ("product.c" if target == "c" else "product")
                            jobs.append((index, output, self.start(source, output, scratch, "--target", target)))
                        self.assertGreater(sum(process.poll() is None for _, _, process in jobs), 1,
                                           "I require live overlapping compiler processes for this check")
                        for index, output, process in jobs:
                            status, diagnostics = self.finish(process)
                            self.assertEqual(status, 0, diagnostics)
                            binary = output
                            if target == "c":
                                binary = output.with_suffix("")
                                result = subprocess.run(["cc", "-std=gnu11", "-I", str(ROOT / "src"),
                                                         "-I", str(ROOT / "modules/std"), str(output),
                                                         "-lm", "-o", str(binary)], capture_output=True, timeout=30)
                                self.assertEqual(result.returncode, 0, result.stderr)
                            result = subprocess.run([str(binary)], capture_output=True, timeout=5)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertEqual(result.stdout, f"{index + 11}\n".encode())
                    finally:
                        for _, _, process in jobs:
                            if process.poll() is None:
                                os.killpg(process.pid, signal.SIGKILL)
                            process.communicate()
                    self.assertEqual(list(scratch.iterdir()), [])

    def test_legacy_scratch_symlinks_are_untouched(self):
        with tempfile.TemporaryDirectory(prefix="nano-isolation-") as tmp:
            directory = Path(tmp)
            scratch = directory / "scratch"
            scratch.mkdir()
            protected = directory / "preserve"
            protected.write_bytes(b"not compiler output")
            for name in ("nanolang_merge_tmp.nano", "nanolang_merged.nano", "merged_debug.nano", "nanolang_temp.c"):
                (scratch / name).symlink_to(protected)
            original = sorted(path.name for path in scratch.iterdir())
            source = self.source(directory / "source", 0)
            status, diagnostics = self.finish(self.start(source, source.with_suffix(".c"), scratch, "--target", "c"))
            self.assertEqual(status, 0, diagnostics)
            self.assertEqual(protected.read_bytes(), b"not compiler output")
            self.assertEqual(sorted(path.name for path in scratch.iterdir()), original)

    def test_source_only_needs_no_temporary_directory(self):
        with tempfile.TemporaryDirectory(prefix="nano-isolation-") as tmp:
            directory = Path(tmp)
            source = self.source(directory / "source", 2)
            scratch = directory / "absent" / "scratch"
            output = source.with_suffix(".c")
            status, diagnostics = self.finish(self.start(source, output, scratch, "--target", "c", cc="/bin/false"))
            self.assertEqual(status, 0, diagnostics)
            self.assertIn(b"return 13", output.read_bytes())
            self.assertFalse(scratch.parent.exists())

    def test_failures_leave_no_shared_source_or_private_staging(self):
        sources = ["fn main() -> int {", "fn main() -> int { return missing }",
                   "fn main() -> int { return 0 } shadow main { assert false }",
                   "fn main() -> int { return 0 }"]
        with tempfile.TemporaryDirectory(prefix="nano-isolation-") as tmp:
            directory = Path(tmp)
            scratch = directory / "scratch"
            scratch.mkdir()
            source = directory / "input.nano"
            output = directory / "program"
            for index, text in enumerate(sources):
                with self.subTest(index=index):
                    source.write_text(text)
                    output.write_bytes(b"preserve")
                    status, diagnostics = self.finish(self.start(source, output, scratch,
                                                                  cc="/bin/false" if index == 3 else None))
                    self.assertNotEqual(status, 0, diagnostics)
                    self.assertEqual(output.read_bytes(), b"preserve")
                    self.assertEqual(list(scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
