"""I exercise real GNU-as file reads after macro expansion, outside production."""
import json
import os
import select
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "linux", "I test the Linux dynamic-loader boundary here")
class AssemblerSnapshotTrial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler, cls.assembler = shutil.which("cc"), shutil.which("as")
        if not cls.compiler or not cls.assembler:
            raise unittest.SkipTest("I need a C compiler and GNU as")
        version = subprocess.run([cls.assembler, "--version"], capture_output=True, check=True).stdout
        if b"GNU assembler" not in version:
            raise unittest.SkipTest("I test GNU as, not an arbitrary assembler")
        cls.workspace = tempfile.TemporaryDirectory(prefix="nano-as-open-library-")
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.preload = Path(cls.workspace.name) / "capture.so"
        instrumentation = ["-fsanitize=undefined", "-fno-sanitize-recover=all"] if os.getenv("NANO_AS_TRIAL_UBSAN") else []
        subprocess.run([cls.compiler, "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-O2",
            *instrumentation, str(ROOT / "tests/fixtures/assembler_snapshot_preload.c"), "-ldl", "-o", str(cls.preload)],
            capture_output=True, check=True, timeout=30)

    def run_tool(self, argv, directory, env=None, success=True):
        result = subprocess.run([str(arg) for arg in argv], cwd=directory, env=env,
                                capture_output=True, timeout=20)
        if success: self.assertEqual(result.returncode, 0, result.stderr)
        else: self.assertNotEqual(result.returncode, 0, result.stderr)
        return result

    def records(self, directory):
        data = (directory / "reads").read_bytes()
        result = []
        while data:
            self.assertGreaterEqual(len(data), 8)
            length, index = struct.unpack("=II", data[:8])
            self.assertGreaterEqual(len(data), 8 + length)
            path = os.fsdecode(data[8:8 + length])
            result.append((path, (directory / f"input-{index}").read_bytes()))
            data = data[8 + length:]
        return result

    def test_expanded_reads_replay_without_originals(self):
        for spelling in ("plain", "spaces ' $ #", "back\\slash", "line\nbreak"):
            with self.subTest(spelling=spelling), tempfile.TemporaryDirectory(prefix="nano-as-open-trial-") as tmp:
                directory = Path(tmp)
                snapshots = directory / "snapshots"
                snapshots.mkdir()
                binary = directory / (spelling + ".bin")
                binary.write_bytes(b"xx42yy")
                include = directory / "macro.s"
                # I let the assembler resolve a macro-argument filename.
                filename = str(binary).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                include.write_text('.macro payload file\n.incbin "\\file", (1+1), (3-1)\n.endm\n'
                                   f'payload "{filename}"\n')
                source, assembly = directory / "answer.c", directory / "answer.s"
                inline = '.data\n.globl snapshot_payload\nsnapshot_payload:\n.include "macro.s"\n.text\n'
                source.write_text('extern const unsigned char snapshot_payload[];\n__asm__(' + json.dumps(inline) + ');\n'
                    'long long nano_build_answer(void) { return (snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
                self.run_tool([self.compiler, "-fPIC", "-O2", "-g", "-S", source, "-o", assembly], directory)
                env = os.environ.copy()
                env.update(LD_PRELOAD=str(self.preload), NANO_AS_TRIAL_DIRECTORY=str(snapshots),
                           NANO_AS_TRIAL_PHASE="capture")
                first, replay = directory / "first.o", directory / "replay.o"
                self.run_tool([self.assembler, assembly, "-o", first], directory, env)
                captured = dict(self.records(snapshots))
                self.assertEqual(captured[str(binary)], b"xx42yy")
                self.assertEqual(captured["macro.s"], include.read_bytes())
                self.assertEqual(captured[str(assembly)], assembly.read_bytes())
                env["NANO_AS_TRIAL_PHASE"] = "replay"
                binary.write_bytes(b"xx43yy")
                include.write_text('.ascii "44"\n')
                self.run_tool([self.assembler, assembly, "-o", replay], directory, env)
                self.assertEqual(first.read_bytes(), replay.read_bytes())
                for original in (source, assembly, include, binary): original.unlink()
                self.run_tool([self.assembler, assembly, "-o", replay], directory, env)
                self.assertEqual(first.read_bytes(), replay.read_bytes())
                library = directory / "answer.so"
                self.run_tool([self.compiler, "-shared", replay, "-o", library], directory)
                result = self.run_tool([sys.executable, "-c", "import ctypes,sys; "
                    "lib=ctypes.CDLL(sys.argv[1]); lib.nano_build_answer.restype=ctypes.c_int64; "
                    "print(lib.nano_build_answer())", library], directory)
                self.assertEqual(result.stdout.strip(), b"42")
                unrecorded = directory / "unrecorded.s"
                unrecorded.write_text(".text\n")
                self.run_tool([self.assembler, unrecorded, "-o", replay], directory, env, success=False)

    def test_repeated_path_keeps_each_captured_read(self):
        with tempfile.TemporaryDirectory(prefix="nano-as-repeated-read-") as tmp:
            directory = Path(tmp)
            snapshots = directory / "snapshots"
            snapshots.mkdir()
            binary, assembly = directory / "payload.bin", directory / "input.s"
            binary.write_bytes(b"42")
            assembly.write_text('.data\n.globl snapshot_payload\nsnapshot_payload:\n' +
                                f'.incbin "{binary}"\n.incbin "{binary}"\n')
            notify_read, notify_write = os.pipe()
            release_read, release_write = os.pipe()
            env = os.environ.copy()
            env.update(LD_PRELOAD=str(self.preload), NANO_AS_TRIAL_DIRECTORY=str(snapshots),
                       NANO_AS_TRIAL_PHASE="capture", NANO_AS_TRIAL_PAUSE_PATH=str(binary),
                       NANO_AS_TRIAL_NOTIFY_FD=str(notify_write), NANO_AS_TRIAL_RELEASE_FD=str(release_read))
            first, replay = directory / "first.o", directory / "replay.o"
            child = None
            try:
                child = subprocess.Popen([self.assembler, str(assembly), "-o", str(first)], cwd=directory,
                    env=env, pass_fds=(notify_write, release_read), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertTrue(select.select([notify_read], [], [], 10)[0], "I did not reach the read barrier")
                self.assertEqual(os.read(notify_read, 1), b"1")
                binary.write_bytes(b"43")
                os.write(release_write, b"1")
                _, error = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, error)
            finally:
                for fd in (notify_read, notify_write, release_read, release_write): os.close(fd)
                if child and child.poll() is None:
                    child.kill()
                    child.communicate()
            reads = [data for name, data in self.records(snapshots) if name == str(binary)]
            self.assertEqual(reads, [b"42", b"43"])
            binary.unlink()
            assembly.unlink()
            env["NANO_AS_TRIAL_PHASE"] = "replay"
            self.run_tool([self.assembler, assembly, "-o", replay], directory, env)
            self.assertEqual(first.read_bytes(), replay.read_bytes())
            library = directory / "answer.so"
            self.run_tool([self.compiler, "-shared", replay, "-o", library], directory)
            result = self.run_tool([sys.executable, "-c", "import ctypes,sys; "
                "lib=ctypes.CDLL(sys.argv[1]); print((ctypes.c_char*4).in_dll(lib,'snapshot_payload').raw.decode())",
                library], directory)
            self.assertEqual(result.stdout.strip(), b"4243")
            # I do not recover a missing snapshot by opening its live original.
            binary.write_bytes(b"99")
            (snapshots / "input-1").unlink()
            self.run_tool([self.assembler, assembly, "-o", replay], directory, env, success=False)


if __name__ == "__main__":
    unittest.main()
