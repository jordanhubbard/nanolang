"""I require complete captures and completed replay at the GNU-as stdio boundary."""
import errno
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux", "I exercise a Linux loader helper")
class AssemblerCaptureRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler, cls.assembler = shutil.which("cc"), shutil.which("as")
        cls.work = tempfile.TemporaryDirectory(prefix="nano-as-record-library-")
        cls.addClassCleanup(cls.work.cleanup)
        cls.helper = Path(os.getenv("NANO_AS_CAPTURE_TEST_HELPER", str(Path(cls.work.name) / "capture.so")))
        flags = ["-fsanitize=undefined", "-fno-sanitize-recover=all"] if os.getenv("NANO_AS_TRIAL_UBSAN") else []
        if not os.getenv("NANO_AS_CAPTURE_TEST_HELPER"):
            subprocess.run([cls.compiler, "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-O2", *flags,
                            str(ROOT / "src/runtime/assembler_capture.c"), "-ldl", "-o", str(cls.helper)],
                           capture_output=True, check=True, timeout=30)
        source = Path(cls.work.name) / "reads.c"
        source.write_text('''#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    char bytes[2];
    FILE *file = fopen(argv[2], "rb");
    if (!file || fread(bytes, 1, 2, file) != 2 || memcmp(bytes, "42", 2)) return 3;
    fclose(file);
    if (!strcmp(argv[1], "short")) return 0;
    if (!strcmp(argv[1], "killed")) { raise(SIGKILL); return 4; }
    errno = 0;
    file = fopen(argv[3], "rb");
    if (file || errno != ENOENT) return 5;
    int fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || write(fd, !strcmp(argv[1], "capture") ? "43" : "99", 2) != 2) return 6;
    close(fd);
    file = fopen(argv[2], "rb");
    if (!file || fread(bytes, 1, 2, file) != 2 || memcmp(bytes, "43", 2)) return 7;
    fclose(file);
    return 0;
}
''')
        cls.reader = Path(cls.work.name) / "reads"
        subprocess.run([cls.compiler, "-Wall", "-Wextra", "-Werror", source, "-o", cls.reader],
                       capture_output=True, check=True, timeout=30)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nano-as-records-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.prefix = self.directory / "capture"
        self.input = self.directory / "input.s"
        self.binary = self.directory / "payload.bin"
        self.binary.write_bytes(b"42")
        self.input.write_text(f'.data\n.incbin "{self.binary}"\n')
        self.environment = os.environ.copy()
        self.environment.update(LD_PRELOAD=str(self.helper), NANO_AS_CAPTURE_PREFIX=str(self.prefix),
                                NANO_AS_CAPTURE_INPUT=str(self.input))

    def invoke(self, phase, command=None, success=True):
        env = dict(self.environment, NANO_AS_CAPTURE_PHASE=phase)
        result = subprocess.run(command or [self.assembler, self.input, "-o", self.directory / "output.o"],
                                cwd=self.directory, env=env, capture_output=True, timeout=10)
        if success: self.assertEqual(result.returncode, 0, result.stderr)
        else: self.assertNotEqual(result.returncode, 0, result.stderr)
        return result

    def test_real_assembler_sealed_replay(self):
        include = self.directory / "macro.s"
        include.write_text(f'.macro payload file\n.incbin "\\file"\n.endm\npayload "{self.binary}"\n')
        self.input.write_text(f'.data\n.include "{include}"\n')
        self.invoke("capture")
        original = (self.directory / "output.o").read_bytes()
        self.assertFalse(Path(str(self.prefix) + ".partial0").exists())
        for path in (self.input, include, self.binary): path.unlink()
        self.invoke("replay")
        self.assertEqual((self.directory / "output.o").read_bytes(), original)
        self.assertEqual(Path(str(self.prefix) + ".replayed0").read_bytes(), b"NACDONE1")

    def test_repeated_and_failed_opens_are_ordered(self):
        self.environment["NANO_AS_CAPTURE_INPUT"] = str(self.binary)
        missing = self.directory / "missing"
        self.invoke("capture", [self.reader, "capture", self.binary, missing])
        manifest = Path(str(self.prefix) + ".manifest0").read_bytes()
        self.assertEqual(manifest[:8], b"NASCAP01")
        at, observed = 8, []
        while True:
            kind, length, error, reserved, size, checksum = struct.unpack("<IIIIQQ", manifest[at:at + 32])
            at += 32
            if not kind: break
            observed.append((manifest[at:at + length].decode(), error, size))
            at += length
        self.assertEqual(observed, [(str(self.binary), 0, 2), (str(missing), errno.ENOENT, 0), (str(self.binary), 0, 2)])
        self.binary.unlink()
        missing.write_bytes(b"I must remain invisible during replay")
        self.invoke("replay", [self.reader, "replay", self.binary, missing])
        self.assertEqual(self.binary.read_bytes(), b"99")
        self.invoke("replay", [self.reader, "short", self.binary, missing], success=False)
        self.assertEqual(Path(str(self.prefix) + ".replayed0").read_bytes(), b"")

    def test_search_macro_and_auxiliary_modes_replay_without_originals(self):
        for mode, flags in (("ordinary", []), ("debug-listing", ["-g", "-alh"]),
                            ("alternate-macros", ["--alternate"])):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=self.directory) as work:
                directory = Path(work)
                early, included = directory / "early", directory / "included"
                early.mkdir()
                included.mkdir()
                source = directory / "input.s"
                nested = included / "nested.s"
                macro = included / "macro.s"
                payload = included / "payload.bin"
                source.write_text('.data\n.include "nested.s"\n')
                nested.write_text('.include "macro.s"\n.rept 2\nemit\n.endr\n'
                                  '.if 0\n.include "inactive.s"\n.endif\n')
                macro.write_text('.set start, 1\n.set count, 2\n.macro emit\n'
                                 f'.incbin "{payload}", start, count\n.endm\n')
                payload.write_bytes(b"x42y")
                prefix = directory / "record"
                self.environment.update(NANO_AS_CAPTURE_PREFIX=str(prefix),
                                        NANO_AS_CAPTURE_INPUT=str(source))
                output, dependencies = directory / "output.o", directory / "output.d"
                command = [self.assembler, *flags, "-I", early, "-I", included,
                           "--MD", dependencies, source, "-o", output]
                captured = self.invoke("capture", command)
                original, original_deps = output.read_bytes(), dependencies.read_bytes()
                self.assertIn(str(source).encode(), original_deps)
                if mode == "debug-listing": self.assertTrue(captured.stdout)
                # I require the payload twice, not merely an object that happens
                # to remain unchanged after a silently skipped directive.
                raw = directory / "data.bin"
                subprocess.run(["objcopy", "--dump-section", f".data={raw}", output],
                               capture_output=True, check=True, timeout=10)
                self.assertEqual(raw.read_bytes(), b"4242")
                for path in (source, nested, macro, payload): path.unlink()
                for name in ("nested.s", "macro.s"):
                    (early / name).write_text('.error "I must not read this new candidate"\n')
                output.unlink()
                dependencies.unlink()
                replayed = self.invoke("replay", command)
                self.assertEqual(output.read_bytes(), original)
                self.assertEqual(dependencies.read_bytes(), original_deps)
                self.assertEqual(replayed.stdout, captured.stdout)
                self.assertEqual(replayed.stderr, captured.stderr)
                self.assertEqual(Path(str(prefix) + ".replayed0").read_bytes(), b"NACDONE1")

    def test_partial_capture_cannot_replay(self):
        self.environment["NANO_AS_CAPTURE_INPUT"] = str(self.binary)
        missing = self.directory / "missing"
        self.invoke("capture", [self.reader, "killed", self.binary, missing], success=False)
        self.assertTrue(Path(str(self.prefix) + ".partial0").is_file())
        self.assertFalse(Path(str(self.prefix) + ".manifest0").exists())
        self.invoke("replay", [self.reader, "short", self.binary, missing], success=False)

    def test_corruption_and_copy_substitution_fail(self):
        self.invoke("capture")
        manifest = Path(str(self.prefix) + ".manifest0")
        original = manifest.read_bytes()
        variants = [original[:i] for i in (0, 7, 8, 31, len(original) - 1)]
        variants += [original + b"extra", b"WRONG001" + original[8:],
                     original[:12] + struct.pack("<I", 0xffffffff) + original[16:],
                     original[:-1] + bytes([original[-1] ^ 1])]
        for variant in variants:
            with self.subTest(length=len(variant)):
                manifest.write_bytes(variant)
                self.invoke("replay", success=False)
        manifest.write_bytes(original)
        copy = Path(str(self.prefix) + ".input1")
        copy.chmod(0o600)
        copy.write_bytes(b"43")
        self.invoke("replay", success=False)
        copy.unlink()
        self.invoke("replay", success=False)
        copy.symlink_to(self.binary)
        self.invoke("replay", success=False)

    def test_existing_capture_is_not_overwritten(self):
        self.invoke("capture")
        manifest = Path(str(self.prefix) + ".manifest0").read_bytes()
        self.invoke("capture", success=False)
        self.assertEqual(Path(str(self.prefix) + ".manifest0").read_bytes(), manifest)

    def test_source_symlink_is_captured_but_fifo_is_rejected(self):
        target = self.directory / "target.bin"
        self.binary.rename(target)
        self.binary.symlink_to(target)
        self.invoke("capture")
        self.binary.unlink()
        os.mkfifo(self.binary)
        self.invoke("replay")
        self.prefix = self.directory / "fifo-capture"
        self.environment["NANO_AS_CAPTURE_PREFIX"] = str(self.prefix)
        self.invoke("capture", success=False)
        self.assertFalse(Path(str(self.prefix) + ".manifest0").exists())

    def test_expected_input_is_required_and_checked(self):
        self.environment["NANO_AS_CAPTURE_INPUT"] = str(self.binary)
        self.invoke("capture", success=False)
        self.assertFalse(Path(str(self.prefix) + ".manifest0").exists())
        del self.environment["NANO_AS_CAPTURE_INPUT"]
        self.invoke("capture", success=False)

    def test_path_bytes_survive_the_sealed_format(self):
        unusual = self.directory / 'payload " \' $ # \\\n.bin'
        self.binary.rename(unusual)
        spelling = str(unusual).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        self.input.write_text(f'.data\n.incbin "{spelling}"\n')
        self.invoke("capture")
        original = (self.directory / "output.o").read_bytes()
        unusual.unlink()
        self.input.unlink()
        self.invoke("replay")
        self.assertEqual((self.directory / "output.o").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
