"""I retain the types and headers needed by imported C declarations."""
from pathlib import Path
import os
import json
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ImportedCallbackTypedefs(unittest.TestCase):
    def test_global_storage_keeps_source_values_across_header_names(self):
        with tempfile.TemporaryDirectory(prefix="nano-header-global-storage-") as directory:
            work = Path(directory)
            source = self.own_header_fixture(work, ['SDL.h'])
            module = work / 'bridge'
            with (module / 'SDL.h').open('a') as header:
                header.write('\n#define HEADER_MACRO 900\nenum { HEADER_ENUM = 901 };\n')
            (module / 'bridge.nano').write_text('''module bridge
pub let HEADER_MACRO: int = 41
let mut HEADER_ENUM: int = 42
let mut private_count: int = 0
pub let mut public_count: int = 0
fn advance()->int { set private_count (+ private_count 1) return private_count }
shadow advance { let saved: int = private_count set private_count 0 assert (== (advance) 1) set private_count saved }
pub fn storage()->int { let saved: int = HEADER_ENUM set HEADER_ENUM (+ HEADER_ENUM 1) let observed: int = HEADER_ENUM set HEADER_ENUM saved return observed }
shadow storage { assert (== (storage) 43) }
pub let derived: int = (+ HEADER_MACRO HEADER_ENUM)
pub fn read()->int { return (+ HEADER_ENUM HEADER_MACRO) }
shadow read { assert (== (read) 83) }
pub fn bump()->int { set public_count (+ public_count (advance)) return public_count }
shadow bump { let saved: int = public_count let prior: int = private_count set public_count 0 set private_count 0 assert (== (bump) 1) set public_count saved set private_count prior }
pub fn lexical(HEADER_ENUM: int)->int { let local: int = HEADER_ENUM return local }
shadow lexical { assert (== (lexical 7) 7) }
pub fn local_value()->int { let HEADER_ENUM: int = 8 return HEADER_ENUM }
shadow local_value { assert (== (local_value) 8) }
struct Record { HEADER_ENUM: int }
pub fn field_value()->int { let item: Record = Record { HEADER_ENUM: 9 } return item.HEADER_ENUM }
shadow field_value { assert (== (field_value) 9) }
''')
            (work / 'other.nano').write_text('pub let HEADER_MACRO: int = 17\n')
            source.write_text('''module "bridge/bridge.nano" as bridge
module "other.nano" as other
fn main()->int {
 assert (== bridge.HEADER_MACRO 41)
 assert (== other.HEADER_MACRO 17)
 assert (== bridge.derived 83)
 assert (== (bridge.storage) 43)
 assert (== (bridge.read) 83)
 assert (== (bridge.lexical 9) 9)
 assert (== (bridge.local_value) 8)
 assert (== (bridge.field_value) 9)
 set bridge.public_count 10
 assert (== (bridge.bump) 11)
 assert (== (bridge.bump) 13)
 set bridge.public_count 0
 return 0
}
shadow main { assert (== (main) 0) }
''')
            output = work / 'program'
            result = subprocess.run([str(ROOT / 'bin/nanoc_c'), str(source), '--keep-c', '-o', str(output)],
                cwd=work, env=dict(os.environ, NANO_BUILD_CACHE=str(work / 'cache')),
                capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = subprocess.run([str(output)], cwd=work, capture_output=True, timeout=10)
            self.assertEqual((run.returncode, run.stdout, run.stderr), (0, b'', b''))

    def test_header_collision_does_not_relax_global_write_or_visibility(self):
        for statement in ('set bridge.HEADER_MACRO 5', 'assert (== bridge.HEADER_ENUM 42)'):
            with self.subTest(statement=statement), tempfile.TemporaryDirectory(prefix='nano-header-global-refusal-') as directory:
                work = Path(directory)
                source = self.own_header_fixture(work, ['SDL.h'])
                with (work / 'bridge/SDL.h').open('a') as header:
                    header.write('\n#define HEADER_MACRO 900\nenum { HEADER_ENUM = 901 };\n')
                (work / 'bridge/bridge.nano').write_text('module bridge\npub let HEADER_MACRO: int = 41\nlet HEADER_ENUM: int = 42\n')
                source.write_text('module "bridge/bridge.nano" as bridge\nfn main()->int { ' + statement + ' return 0 }\nshadow main { assert (== (main) 0) }\n')
                output = work / 'program'
                output.write_bytes(b'prior-output')
                result = subprocess.run([str(ROOT / 'bin/nanoc_c'), str(source), '-o', str(output)],
                    cwd=work, env=dict(os.environ, NANO_BUILD_CACHE=str(work / 'cache')),
                    capture_output=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b'prior-output')

    def own_header_fixture(self, work, headers):
        module = work / "bridge"
        module.mkdir()
        (module / "SDL.h").write_text("#ifndef NANO_OWN_HEADER_SDL_H\n"
            "#define NANO_OWN_HEADER_SDL_H\ntypedef unsigned int Uint32;\n"
            "Uint32 SDL_GetTicks(void);\n#endif\n")
        (module / "anchor.c").write_text('#include "SDL.h"\nUint32 SDL_GetTicks(void) { return 42; }\n')
        (module / "bridge.nano").write_text('module bridge\nextern fn SDL_GetTicks() -> int\n'
            'pub fn answer()->int{return 42}\nshadow answer{assert (== (answer) 42)}\n')
        (module / "module.json").write_text(json.dumps({'name': 'bridge', 'headers': headers,
            'c_sources': ['anchor.c'], 'include_dirs': ['.']}))
        source = work / "main.nano"
        source.write_text('module "bridge/bridge.nano" as bridge\n'
            'fn main()->int{assert (== (bridge.answer) 42) return 0}\n'
            'shadow main{assert (== (main) 0)}\n')
        return source

    def test_native_module_implementation_retains_its_own_header(self):
        with tempfile.TemporaryDirectory(prefix="nano-own-module-header-") as directory:
            work = Path(directory)
            source = self.own_header_fixture(work, ['SDL.h'])
            output = work / "program"
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source), '-o', str(output)],
                cwd=work, env=dict(os.environ, NANO_BUILD_CACHE=str(work / 'cache')),
                capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = subprocess.run([str(output)], cwd=work, capture_output=True, timeout=10)
            self.assertEqual((run.returncode, run.stdout, run.stderr), (0, b'', b''))

    def test_undeclared_native_types_preserve_prior_output_on_refusal(self):
        with tempfile.TemporaryDirectory(prefix="nano-missing-module-header-") as directory:
            work = Path(directory)
            source = self.own_header_fixture(work, [])
            output = work / "program"
            output.write_bytes(b'prior-output')
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source), '-o', str(output)],
                cwd=work, env=dict(os.environ, NANO_BUILD_CACHE=str(work / 'cache')),
                capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(b'Uint32', result.stdout + result.stderr)
            self.assertEqual(output.read_bytes(), b'prior-output')

    def test_implicit_public_and_opaque_callback_signatures(self):
        with tempfile.TemporaryDirectory(prefix="nano-callback-types-") as directory:
            work = Path(directory)
            module = work / "callbacks.nano"
            module.write_text("""opaque type Context
opaque type Buffer
extern fn install(callback: fn(Context, Buffer, int) -> void, context: Context) -> int
extern fn select_callback(callback: fn(int) -> bool) -> void
""")
            source = work / "main.nano"
            source.write_text('module "callbacks.nano" as Callbacks\n'
                              'fn main() -> int { return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            generated = work / "program.c"
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source),
                                     "--keep-c", "-o", str(work / "program")],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = generated.read_text()
            self.assertLess(text.index("typedef void (*FnType_"), text.index("extern int64_t install("))
            self.assertRegex(text, r"typedef void \(\*FnType_\d+\)\(void\*, void\*, int64_t\)")
            checked = subprocess.run([*shlex.split(os.environ.get("CC", "cc")),
                                      "-std=c11", "-Werror=incompatible-pointer-types",
                                      "-fsyntax-only", "-I", str(ROOT / "src"), str(generated)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


if __name__ == "__main__":
    unittest.main()
