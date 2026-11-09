"""I execute my actual checked collection boundary in assertion and sanitizer builds."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CollectionHandles(unittest.TestCase):
    def test_actual_native_lifetime(self):
        work = Path(tempfile.mkdtemp(prefix="collection-handles-", dir=ROOT / ".test_output"))
        print("I retain collection handle evidence at", work, flush=True)
        cc = shlex.split(os.environ.get("CC", "cc"))
        compiler = Path(shutil.which(cc[0]) or cc[0]).resolve(strict=True)
        compiler_hash = hashlib.sha256(compiler.read_bytes()).hexdigest()
        version = subprocess.check_output([*cc, "--version"], text=True, timeout=10)
        inputs = ["tests/test_collection_handles.py", "modules/std/collections/module.json",
                  "tests/test_collection_handles.c", "modules/std/collections/collections.c",
                  "modules/std/collections/collections.h", "src/runtime/dyn_array.c",
                  "src/runtime/gc.c", "src/runtime/gc_struct.c"]
        hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs}
        operations = []
        for name, flags in (("plain", []), ("sanitized", ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]), ("threaded", ["-fsanitize=thread"])):
            binary = work / name
            build = [*cc, "-std=c99", "-g", "-O1",
                     "-Wall", "-Wextra", "-Werror", "-UNDEBUG", "-pthread", *flags, "-Isrc",
                     "tests/test_collection_handles.c", "src/runtime/dyn_array.c", "src/runtime/gc.c",
                     "src/runtime/gc_struct.c", "-lm", "-o", str(binary)]
            for label, command in (("build-" + name, build), ("run-" + name, [str(binary)])):
                result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
                (work / (label + ".stdout")).write_bytes(result.stdout)
                (work / (label + ".stderr")).write_bytes(result.stderr)
                operations.append({"label": label, "command": command, "returncode": result.returncode})
                (work / "operations.json").write_text(json.dumps(operations, indent=2) + "\n")
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                if label.startswith("build-"):
                    before = hashlib.sha256(binary.read_bytes()).hexdigest()
                    self.assertIn(binary.read_bytes()[:4], (b"\xcf\xfa\xed\xfe", b"\x7fELF"))
                else:
                    self.assertEqual(hashlib.sha256(binary.read_bytes()).hexdigest(), before)
                    (work / (name + ".identity.json")).write_text(json.dumps({"sha256_before_after": before}) + "\n")
        self.assertEqual(hashes, {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs})
        (work / "inputs-before-after.json").write_text(json.dumps(hashes, indent=2) + "\n")
        self.assertEqual(hashlib.sha256(compiler.read_bytes()).hexdigest(), compiler_hash)
        (work / "compiler.json").write_text(json.dumps({"argv": cc, "observed_launcher": str(compiler),
            "sha256_before_after": compiler_hash, "version": version,
            "scope": "I retain selected input and compiler-launcher identities, not complete header or toolchain closure."}, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
