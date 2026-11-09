"""I test native array copy-back and allocation failures with real heap code."""
import os
import json
import hashlib
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FfiArrayCopyback(unittest.TestCase):
    def test_frame(self):
        work = Path(tempfile.mkdtemp(prefix="ffi-array-results-", dir=ROOT / ".test_output"))
        print("I retain real array-frame evidence at", work, flush=True)
        operations = []
        inputs = ["tests/nanovm/test_ffi_arrays.c", "src/nanovm/vm_ffi_arrays.c",
                  "src/nanovm/vm_ffi_arrays.h", "modules/std/collections/collections.c"]
        hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
        for name, flags in (("plain", []), ("sanitized", ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"])):
            output = work / name
            command = [*shlex.split(os.environ.get("CC", "cc")),
                "-std=c99", "-g", "-O1", "-Wall", "-Wextra", "-Werror", "-UNDEBUG", "-pthread", "-Isrc", *flags,
                "tests/nanovm/test_ffi_arrays.c", "src/nanovm/heap.c", "src/nanovm/value.c",
                "src/nanovm/heap_cycles.c", "src/nanoisa/isa.c", "src/runtime/dyn_array.c",
                "src/runtime/gc.c", "src/runtime/gc_struct.c", "src/utf8.c",
                "modules/std/collections/collections.c", "-lm", "-o", str(output)]
            for label, argv in (("build-"+name, command), ("run-"+name, [str(output)])):
                result = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=60)
                (work/(label+".stdout")).write_bytes(result.stdout)
                (work/(label+".stderr")).write_bytes(result.stderr)
                operations.append({"label":label,"argv":argv,"returncode":result.returncode})
                (work/"operations.json").write_text(json.dumps(operations,indent=2)+"\n")
                self.assertEqual(result.returncode,0,result.stderr.decode(errors="replace"))
                if label.startswith("build-"):
                    before = hashlib.sha256(output.read_bytes()).hexdigest()
                    self.assertIn(output.read_bytes()[:4], (b"\xcf\xfa\xed\xfe", b"\x7fELF"))
                else:
                    self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),before)
                    (work/(name+".identity.json")).write_text(json.dumps({"sha256_before_after":before})+"\n")
        self.assertEqual(hashes,{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs})
        (work/"inputs-before-after.json").write_text(json.dumps(hashes,indent=2)+"\n")



if __name__ == "__main__":
    unittest.main()
