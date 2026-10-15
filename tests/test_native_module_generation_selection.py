"""I select one foreign generation per module directory and native link."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANO_NATIVE_MODULE_COMPILER", ROOT / "bin/nanoc_c"))


class NativeModuleGenerationSelection(unittest.TestCase):
    def test_transitive_dependency_header_paths_reach_module_compilation(self):
        with tempfile.TemporaryDirectory(prefix="nano-dep-headers-") as tmp:
            work = Path(tmp)

            # A leaf dependency exposes its header directory through module.json.
            bridge = work / "bridge"
            bridge_include = bridge / "include"
            bridge_include.mkdir(parents=True)
            (bridge_include / "bridge_value.h").write_text(
                "#include <stdint.h>\n#define BRIDGE_BASE 40\n")
            (bridge / "module.json").write_text(json.dumps({
                "name": "bridge", "headers": ["bridge_value.h"],
                "include_dirs": [str(bridge_include)]}))

            # A middle dependency names the bridge and includes its header.
            middle = work / "middle"
            middle.mkdir()
            (middle / "middle.c").write_text(
                "#include <bridge_value.h>\n"
                "int64_t middle_value(void) { return BRIDGE_BASE + 1; }\n")
            (middle / "middle_api.nano").write_text("pub extern fn middle_value() -> int\n")
            (middle / "module.json").write_text(json.dumps({
                "name": "middle", "c_sources": ["middle.c"], "dependencies": ["bridge"]}))

            # A second dependency exposes its header directory through cflags.
            flags = work / "flags"
            flags_include = flags / "api"
            flags_include.mkdir(parents=True)
            (flags_include / "flags_value.h").write_text(
                "#include <stdint.h>\n#define FLAGS_BASE 10\n")
            (flags / "module.json").write_text(json.dumps({
                "name": "flags", "headers": ["flags_value.h"],
                "cflags": ["-I" + str(flags_include)]}))

            # The top module reaches both headers through its declared
            # dependencies: one directly, the other only transitively.
            top = work / "top"
            top.mkdir()
            (top / "top.c").write_text(
                "#include <bridge_value.h>\n"
                "#include <flags_value.h>\n"
                "int64_t top_value(void) { return BRIDGE_BASE + FLAGS_BASE + 2; }\n")
            (top / "top_api.nano").write_text("pub extern fn top_value() -> int\n")
            (top / "module.json").write_text(json.dumps({
                "name": "top", "c_sources": ["top.c"],
                "dependencies": ["middle", "flags"]}))

            source = work / "main.nano"
            source.write_text(
                'module "middle/middle_api.nano" as middle\n'
                'module "top/top_api.nano" as top\n'
                'fn main() -> int { unsafe { assert (== (middle.middle_value) 41) '
                'assert (== (top.top_value) 52) } return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            env = os.environ.copy()
            for name in ("NANO_CC", "CC"):
                env.pop(name, None)
            env["NANO_BUILD_CACHE"] = str(work / "cache")
            result = subprocess.run(
                [str(COMPILER), str(source), "-o", str(work / "program")],
                cwd=ROOT, env=env, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            subprocess.run([str(work / "program")], check=True, timeout=10)

    def test_pure_nano_dependency_keeps_foreign_headers(self):
        with tempfile.TemporaryDirectory(prefix='nano-pure-import-headers-') as tmp:
            work = Path(tmp)
            foreign = work/'foreign'
            include = foreign/'include directory'
            include.mkdir(parents=True)
            (include/'foreign_value.h').write_text('#include <stdint.h>\nint64_t foreign_value(void);\n')
            (foreign/'value.c').write_text('#include "foreign_value.h"\nint64_t foreign_value(void){return 42;}\n')
            (foreign/'api.nano').write_text('pub extern fn foreign_value()->int\n')
            (foreign/'module.json').write_text(json.dumps({'name':'foreign',
                'headers':['foreign_value.h'], 'include_dirs':[str(include)], 'c_sources':['value.c']}))
            (work/'middle.nano').write_text('module "foreign/api.nano" as foreign\n'
                'pub fn answer()->int {unsafe {return (foreign.foreign_value)}}\n'
                'shadow answer {assert (== (answer) 42)}\n')
            source, binary = work/'main.nano', work/'program'
            source.write_text('module "middle.nano" as middle\n'
                'fn main()->int{assert (== (middle.answer) 42) return 0}\n'
                'shadow main {assert (== (main) 0)}\n')
            env = {**os.environ, 'NANO_BUILD_CACHE':str(work/'cache')}
            result = subprocess.run([ROOT/'bin/nanoc_c',source,'-o',binary],
                cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            subprocess.run([binary],check=True,timeout=10)

    def test_long_link_closure_keeps_every_foreign_object(self):
        with tempfile.TemporaryDirectory(prefix="nano-long-link-") as tmp:
            work = Path(tmp)
            imports = []
            calls = []
            for index in range(8):
                module = work / f"foreign{index}"
                module.mkdir()
                name = f"link_value_{index}"
                (module / "module.json").write_text(json.dumps({
                    "name": f"foreign{index}", "c_sources": ["foreign.c"],
                }))
                (module / "foreign.c").write_text(
                    f"#include <stdint.h>\nint64_t {name}(void) {{ return {index + 1}; }}\n")
                (module / f"api{index}.nano").write_text(f"pub extern fn {name}() -> int\n")
                imports.append(f'unsafe module "foreign{index}/api{index}.nano" as m{index}')
                calls.append(f"assert (== (m{index}.{name}) {index + 1})")
            source = work / "main.nano"
            source.write_text("\n".join(imports) + "\nfn main() -> int { unsafe { " +
                              " ".join(calls) + " } return 0 }\nshadow main { assert true }\n")
            env = os.environ.copy()
            env.pop("NANO_CC", None)
            env.pop("CC", None)
            cache = work / ("cache_" + "x" * 150) / ("nested_" + "y" * 150)
            env["NANO_BUILD_CACHE"] = str(cache)
            # I exercise both initial publication and the cached generation path.
            for _ in range(2):
                result = subprocess.run(
                    [str(COMPILER), str(source), "-o", str(work / "program")],
                    cwd=ROOT, env=env, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                objects = list(cache.glob("*/.nano-gen-*/*.o"))
                self.assertEqual(len(objects), 8)
                self.assertGreater(sum(len(str(obj)) + 3 for obj in objects), 2048)
                subprocess.run([str(work / "program")], check=True, timeout=10)

    def test_oversized_compile_flags_fail_without_silent_truncation(self):
        with tempfile.TemporaryDirectory(prefix="nano-long-cflags-") as tmp:
            work = Path(tmp)
            module = work / "foreign"
            module.mkdir()
            # A shell-form driver keeps flags inline instead of response transport.
            (module / "module.json").write_text(json.dumps({
                "name": "long_flags", "c_compiler": "cc ", "c_sources": ["foreign.c"],
                "cflags": [f"-DLONG_COMPILE_FLAG_{index:04d}=1" for index in range(120)],
            }))
            (module / "foreign.c").write_text("#include <stdint.h>\nint64_t probe(void) { return 42; }\n")
            (module / "foreign.nano").write_text("pub extern fn probe() -> int\n")
            source = work / "main.nano"
            source.write_text('unsafe module "foreign/foreign.nano" as f\n'
                              'fn main() -> int { unsafe { assert (== (f.probe) 42) } return 0 }\nshadow main { assert true }\n')
            env = os.environ.copy()
            # I exercise the module's shell-form compiler, even under a CI CC override.
            env.pop("NANO_CC", None)
            env.pop("CC", None)
            env["NANO_BUILD_CACHE"] = str(work / "cache")
            result = subprocess.run(
                [str(COMPILER), str(source), "-o", str(work / "program")],
                cwd=ROOT, env=env, capture_output=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"I could not represent all module compile flags.", result.stderr)
            self.assertFalse((work / "program").exists())

    def test_native_dependencies_pin_shared_providers(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-dependencies-") as tmp:
            work = Path(tmp)
            for name in ("provider", "bridge", "first", "second"):
                (work / name).mkdir()
            provider = work / "provider"
            (provider / "common.c").write_text("#include <stdint.h>\nint64_t core_probe(void) { return 21; }\n")
            (provider / "provider.nano").write_text("pub extern fn core_probe() -> int\n")
            (provider / "module.json").write_text(json.dumps({
                "name": "provider", "c_compiler": "cc ", "c_sources": ["common.c"], "dependencies": ["bridge"]}))
            (work / "bridge/module.json").write_text(json.dumps({
                "name": "bridge", "dependencies": ["provider"]}))
            for name in ("first", "second"):
                directory = work / name
                (directory / "entry.c").write_text("#include <stdint.h>\nextern int64_t core_probe(void);\n"
                    f"int64_t {name}_probe(void) {{ return core_probe(); }}\n")
                (directory / "entry.nano").write_text(f"module {name}\npub extern fn {name}_probe() -> int\n")
                (directory / "module.json").write_text(json.dumps({
                    "name": name, "c_sources": ["entry.c"], "dependencies": ["bridge"],
                    "shared_c_sources": ["../provider/common.c"]}))
            source = work / "main.nano"
            source.write_text('unsafe module "first/entry.nano" as a\n'
                'unsafe module "second/entry.nano" as b\n'
                'fn main() -> int { unsafe { assert (== (+ (a.first_probe) (b.second_probe)) 42) '
                '} return 0 }\n'
                'shadow main { assert true }\n')
            env = os.environ.copy()
            env.pop("NANO_CC", None); env.pop("CC", None)
            cache = work / "cache"; env["NANO_BUILD_CACHE"] = str(cache)
            for attempt in range(2):
                if attempt:
                    source.write_text('unsafe module "provider/provider.nano" as provider\n' + source.read_text())
                result = subprocess.run([str(COMPILER), str(source), "-o", str(work / "program")],
                    cwd=ROOT, env=env, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                subprocess.run([str(work / "program")], check=True, timeout=10)
                self.assertEqual(len(list(cache.glob("*/.nano-gen-*/provider.o"))), attempt + 1)
            # I execute each private shared copy independently of native providers.
            import ctypes
            for name in ("first", "second"):
                paths = list(cache.glob(f"*/.nano-gen-*/lib{name}.*"))
                self.assertTrue(paths)
                library = ctypes.CDLL(str(paths[0]))
                self.assertEqual(getattr(library, name + "_probe")(), 21)

    def test_missing_native_dependency_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-missing-native-dependency-") as tmp:
            work = Path(tmp); module = work / "foreign"; module.mkdir()
            (module / "entry.c").write_text("#include <stdint.h>\nint64_t probe(void) { return 42; }\n")
            (module / "entry.nano").write_text("pub extern fn probe() -> int\n")
            (module / "module.json").write_text(json.dumps({"name": "foreign",
                "c_sources": ["entry.c"], "dependencies": ["absent_provider"]}))
            source = work / "main.nano"
            source.write_text('unsafe module "foreign/entry.nano" as f\n'
                'fn main() -> int { unsafe { return (- (f.probe) 42) } }\nshadow main { assert true }\n')
            output = work / "program"; output.write_bytes(b"prior-output")
            result = subprocess.run([str(COMPILER), str(source), "-o", str(output)],
                cwd=ROOT, capture_output=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"I cannot load native dependency", result.stderr)
            self.assertEqual(output.read_bytes(), b"prior-output")

    def test_multiple_sources_share_one_uncacheable_foreign_build(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-generation-") as tmp:
            work = Path(tmp)
            module = work / "foreign"
            module.mkdir()
            # A shell-form compiler deliberately has no reusable tool identity.
            # Every redundant build publishes a different immutable generation.
            (module / "module.json").write_text(json.dumps({
                "name": "foreign", "c_compiler": "cc ",
                "c_sources": ["foreign.c"], "headers": ["foreign.h"],
            }))
            (module / "foreign.h").write_text(
                "#include <stdint.h>\nint64_t first(void);\nint64_t second(void);\n")
            (module / "foreign.c").write_text(
                '#include "foreign.h"\nint64_t first(void) { return 20; }\n'
                'int64_t second(void) { return 22; }\n')
            (module / "first.nano").write_text("pub extern fn first() -> int\n")
            (module / "second.nano").write_text("pub extern fn second() -> int\n")
            source = work / "main.nano"
            source.write_text(
                'unsafe module "foreign/first.nano" as a\n'
                'unsafe module "foreign/second.nano" as b\n'
                'fn main() -> int { unsafe { assert (== (+ (a.first) (b.second)) 42) } return 0 }\n'
                'shadow main { assert true }\n')
            env = os.environ.copy()
            env.pop("NANO_CC", None)
            env.pop("CC", None)
            env["NANO_BUILD_CACHE"] = str(work / "cache")
            for attempt in range(2):
                result = subprocess.run(
                    [str(COMPILER), str(source), "-o", str(work / "program")],
                    cwd=ROOT, env=env, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                subprocess.run([str(work / "program")], check=True, timeout=10)
                generations = list((work / "cache").glob("*/.nano-gen-*"))
                self.assertEqual(len(generations), attempt + 1)


if __name__ == "__main__":
    unittest.main()
