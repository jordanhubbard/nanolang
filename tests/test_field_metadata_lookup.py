"""I resolve field metadata without allocating an index per field read."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]
DRIVER = r'''from "src_nano/typecheck.nano" import FieldMetadata, NSType, TypeKind, lookup_field_type
fn main() -> int {
    let mut metadata: array<FieldMetadata> = []
    let mut i: int = 0
    while (< i 256) {
        set metadata (array_push metadata FieldMetadata { struct_name: "Unused", field_name: (int_to_string i), field_type_kind: TypeKind.TYPE_INT, field_type_is_list: false, field_type_name: "" })
        set i (+ i 1)
    }
    set metadata (array_push metadata FieldMetadata { struct_name: "Parser", field_name: "count", field_type_kind: TypeKind.TYPE_BOOL, field_type_is_list: false, field_type_name: "" })
    set metadata (array_push metadata FieldMetadata { struct_name: "Parser", field_name: "count", field_type_kind: TypeKind.TYPE_INT, field_type_is_list: false, field_type_name: "" })
    set i 0
    while (< i 100) {
        assert (== (lookup_field_type metadata "parser" "count").kind TypeKind.TYPE_INT)
        assert (== (lookup_field_type metadata "Parser" "missing").kind TypeKind.TYPE_UNKNOWN)
        set i (+ i 1)
    }
    return 0
}
shadow main { assert (== (lookup_field_type [] "Missing" "field").kind TypeKind.TYPE_UNKNOWN) }
'''


class FieldMetadataLookup(unittest.TestCase):
    def run_checked(self, args, *, build=False, **kwargs):
        result = run_captured(args, cwd=ROOT, text=True, timeout=180, build=build, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_vm_native_semantics_and_no_per_lookup_map(self):
        with tempfile.TemporaryDirectory(prefix="nano-field-lookup-") as tmp:
            work = Path(tmp)
            source, module, native_c, binary = [work / name for name in
                                               ("lookup.nano", "lookup.nvm", "lookup.c", "lookup")]
            source.write_text(DRIVER)
            self.run_checked([ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", module])
            self.run_checked([ROOT / "bin/nano_vm", module])
            self.run_checked([ROOT / "bin/nvm2c", module, "-o", native_c])
            generated = native_c.read_text()
            marker = "static nmap_t nmap_owned_new(uint8_t kind) {"
            self.assertEqual(generated.count(marker), 1)
            generated = generated.replace(marker, "static size_t lookup_maps, initialization_maps;\n" + marker +
                                          "\n    ++lookup_maps;")
            call = "    int result = (int)nl_main();"
            self.assertEqual(generated.count(call), 1)
            reset = "    initialization_maps = lookup_maps; lookup_maps = 0;\n"
            generated = generated.replace(call, reset + call)
            entry = "int main(int argc, char **argv)"
            self.assertEqual(generated.count(entry), 1)
            generated = generated.replace(entry, "int original_main(int argc, char **argv)")
            generated += r'''
int main(int argc, char **argv) {
    int result = original_main(argc, argv);
    fprintf(stderr, "initialization_maps=%zu lookup_maps=%zu\n", initialization_maps, lookup_maps);
    return result ? result : lookup_maps != 0;
}
'''
            native_c.write_text(generated)
            cc = os.environ.get("NANO_NATIVE_TEST_CC") or shutil.which("cc")
            self.assertTrue(cc)
            self.run_checked([cc, "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror",
                              "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                              native_c, ROOT / "bin/nano_aot_runtime.o", "-lm", "-o", binary,
                              *native_link_flags()], build=True)
            result = self.run_checked([binary], env={**os.environ})
            self.assertIn("lookup_maps=0", result.stderr)
            # I prove that excluding initialization does not hide a guest allocation.
            native_c.write_text(generated.replace(reset, reset + "    (void)nmap_owned_new(1);\n"))
            self.run_checked([cc, "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror",
                              "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                              native_c, ROOT / "bin/nano_aot_runtime.o", "-lm", "-o", binary,
                              *native_link_flags()], build=True)
            rejected = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True,
                                      timeout=180, env={**os.environ})
            self.assertEqual(rejected.returncode, 1, rejected.stdout + rejected.stderr)
            self.assertIn("lookup_maps=1", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
