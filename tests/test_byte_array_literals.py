"""I preserve byte-array storage in contextual literal destinations."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VM = Path(os.environ.get("NANO_VM", ROOT / "bin/nano_vm"))

BYTE_CONTEXTS = {
    "local": "fn main()->int { let bytes:array<u8> = [300] assert (== (at bytes 0) 44) return 0 }",
    "assignment": "fn main()->int { let mut bytes:array<u8> = [] set bytes [300] assert (== (at bytes 0) 44) return 0 }",
    "global": "let bytes:array<u8> = [300] fn main()->int { assert (== (at bytes 0) 44) return 0 }",
    "return": "fn bytes()->array<u8> { return [300] } fn main()->int { assert (== (at (bytes) 0) 44) return 0 }",
    "argument": "fn read(bytes:array<u8>)->int { return (at bytes 0) } fn main()->int { assert (== (read [300]) 44) return 0 }",
    "field": "struct Bytes { data:array<u8> } fn main()->int { let b:Bytes = Bytes { data:[300] } assert (== (at b.data 0) 44) return 0 }",
}

class ByteArrayLiterals(unittest.TestCase):
    def test_contextual_byte_storage(self):
        with tempfile.TemporaryDirectory(prefix="nano-byte-literal-") as tmp:
            for name, source in BYTE_CONTEXTS.items():
                for backend in ("c", "vm"):
                    with self.subTest(name=name, backend=backend):
                        path = Path(tmp) / (name + ".nano")
                        path.write_text(source + " shadow main { assert true }\n")
                        product = path.with_suffix(".nvm" if backend == "vm" else ".out")
                        command = [ROOT / "bin/nano_virt", path, "--emit-nvm"] if backend == "vm" else [ROOT / "bin/nanoc_c", path]
                        compiled = subprocess.run(command + ["-o", product], cwd=ROOT,
                                                  capture_output=True, text=True, timeout=60)
                        self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                        command = [VM, product] if backend == "vm" else [product]
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrong_element_kind_is_refused(self):
        templates = [
            "fn main()->int { let bytes:array<u8> = VALUE return 0 }",
            "fn main()->int { let mut bytes:array<u8> = [] set bytes VALUE return 0 }",
            "let bytes:array<u8> = VALUE fn main()->int { return 0 }",
            "fn bytes()->array<u8> { return VALUE } fn main()->int { return 0 }",
            "fn read(bytes:array<u8>)->int { return 0 } fn main()->int { return (read VALUE) }",
            "struct Bytes { data:array<u8> } fn main()->int { let b:Bytes = Bytes { data:VALUE } return 0 }",
        ]
        with tempfile.TemporaryDirectory(prefix="nano-byte-literal-refusal-") as tmp:
            source = Path(tmp) / "invalid.nano"
            product = Path(tmp) / "output"
            for context, template in enumerate(templates):
                for value in ("[true]", "[1.5]", '["text"]'):
                    for compiler in ("nanoc_c", "nano_virt"):
                        with self.subTest(context=context, value=value, compiler=compiler):
                            source.write_text(template.replace("VALUE", value) + " shadow main { assert true }\n")
                            product.write_bytes(b"prior output")
                            command = [ROOT / "bin" / compiler, source]
                            if compiler == "nano_virt":
                                command += ["--emit-nvm"]
                            result = subprocess.run(command + ["-o", product], cwd=ROOT,
                                                    capture_output=True, text=True, timeout=60)
                            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertEqual(product.read_bytes(), b"prior output")
