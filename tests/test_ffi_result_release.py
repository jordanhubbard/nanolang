"""I load real native images and require result cleanup from the producer's image."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResultRelease(unittest.TestCase):
    def test_same_image(self):
        work = Path(tempfile.mkdtemp(prefix="ffi-result-release-", dir=ROOT / ".test_output"))
        print("I retain actual loader evidence at", work, flush=True)
        dependency = work / "dependency.c"
        dependency.write_text('''#include <stdbool.h>
int dependency_anchor(void) { return 1; }
bool foreign_array__nano_array_release_v1(void *value) { (void)value; return false; }
void foreign_string__nano_string_release_v1(const char *value) { (void)value; }
''')
        origin = work / "origin.c"
        origin.write_text('''#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <string.h>
extern int dependency_anchor(void);
void *foreign_array(void) { (void)dependency_anchor(); abort(); }
void *borrowed_array(void) { return NULL; }
const char *foreign_string(void) { (void)dependency_anchor(); return "rejected"; }
const char *borrowed_string(void) { return "borrowed"; }
const char *owned_string(void) { return strdup("copied"); }
void owned_string__nano_string_release_v1(const char *value) { free((void *)value); }
''')
        extension = work / "snapshot-probe.c"
        extension.write_text('''#include "runtime/dyn_array.h"
#include <stdlib.h>
int64_t snapshot_live(void) { return (int64_t)gc_get_stats().num_objects; }
DynArray *alias_snapshot(DynArray *value) { return value; }
NANO_EXPORT_ARRAY_ABI(alias_snapshot);
bool alias_snapshot__nano_array_release_v1(void *value) { (void)value; abort(); }
''')
        cc = shlex.split(os.environ.get("CC", "cc"))
        crypto = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "openssl"], text=True, timeout=10))
        launcher = Path(shutil.which(cc[0]) or cc[0]).resolve(strict=True)
        launcher_hash = hashlib.sha256(launcher.read_bytes()).hexdigest()
        version = subprocess.check_output([*cc,"--version"],text=True,timeout=10)
        operations, products = [], []
        inputs = [ROOT/p for p in ["tests/test_ffi_result_release.c", "src/runtime/ffi_loader.c",
            "src/runtime/ffi_loader.h", "modules/std/collections/collections.c",
            "src/runtime/dyn_array.c", "src/runtime/gc.c", "src/runtime/gc_struct.c"]] + [dependency, origin, extension]
        before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        for name, sanitizer in (("plain", []), ("sanitized", ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]),
                                ("threaded", ["-fsanitize=thread"])):
            folder = work / name; folder.mkdir()
            ext = "dylib" if sys.platform == "darwin" else "so"
            shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            flags = [*cc, "-std=c99", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-UNDEBUG", "-pthread", "-Isrc", *sanitizer]
            dep, lib, provider = [folder / ("lib"+n+"."+ext) for n in ["dependency", "origin", "collections"]]
            driver = folder / "driver"
            commands = [("dependency", [*flags, *shared, dependency, "-o", dep]),
                ("origin", [*flags, *shared, origin, dep, "-Wl,-rpath,"+str(folder), "-o", lib]),
                ("collections", [*flags, *shared, "modules/std/collections/collections.c",
                    "src/runtime/dyn_array.c", "src/runtime/gc.c", "src/runtime/gc_struct.c", extension, "-lm", "-o", provider]),
                ("driver", [*flags, "tests/test_ffi_result_release.c", "src/runtime/ffi_loader.c",
                    "src/runtime/module_build_dir.c", *crypto, "-ldl", "-o", driver]),
                ("run", [driver, provider, lib])]
            identities = {}
            for label, command in commands:
                command = list(map(str, command)); label = name+"-"+label
                result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
                (work/(label+".stdout")).write_bytes(result.stdout)
                (work/(label+".stderr")).write_bytes(result.stderr)
                operations.append({"label":label,"argv":command,"returncode":result.returncode})
                (work/"operations.json").write_text(json.dumps(operations,indent=2)+"\n")
                self.assertEqual(result.returncode,0,result.stderr.decode(errors="replace"))
                if not label.endswith("-run"):
                    path = Path(command[-1])
                    native = path.read_bytes()
                    if native[:4] == b"\xcf\xfa\xed\xfe":
                        self.assertEqual(struct.unpack_from("<I",native,12)[0],2 if path==driver else 6)
                    else:
                        self.assertEqual(native[:4],b"\x7fELF")
                        kind=struct.unpack_from("<H" if native[5]==1 else ">H",native,16)[0]
                        self.assertIn(kind,(2,3) if path==driver else (3,))
                    identities[str(path)] = hashlib.sha256(native).hexdigest()
            vm = os.environ.get("NANO_SNAPSHOT_VM")
            assembler = os.environ.get("NANO_SNAPSHOT_ASSEMBLER")
            if name == "plain" and vm:
                self.assertTrue(assembler, "I require the explicitly selected fresh assembler too.")
                tools = {path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (vm,assembler)}
                entries = [(provider,"gc_init","void"),(provider,"nl_hm_new","opaque"),
                    (provider,"nl_hm_put","void opaque string string"),(provider,"nl_hm_keys","array opaque"),
                    (provider,"nl_hm_free","void opaque"),(provider,"snapshot_live","int"),(provider,"gc_shutdown","void")]
                imports = "".join(".import "+json.dumps(str(lib))+" "+json.dumps(symbol)+" "+signature+"\n.import_kind "+str(i)+" artifact\n"
                    for i,(lib,symbol,signature) in enumerate(entries))
                positive = imports + ('.string key "owned"\n.string value "value"\n.entry 0\n.function main 0 2 0 int 1\n'
                    'CALL_EXTERN 0\nCALL_EXTERN 1\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_STR key\nPUSH_STR value\nCALL_EXTERN 2\n'
                    'LOAD_LOCAL 0\nCALL_EXTERN 3\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nCALL_EXTERN 4\n'
                    'CALL_EXTERN 5\nPUSH_I64 0\nEQ\nASSERT\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_STR key\nEQ\nASSERT\n'
                    'CALL_EXTERN 6\nPUSH_I64 0\nRET\n.end\n')
                negative = '.import '+json.dumps(str(lib))+' "foreign_array" array\n.import_kind 0 artifact\n.entry 0\n.function main 0 0 0 int 1\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n'
                alias = '.import '+json.dumps(str(provider))+' "alias_snapshot" array array\n.import_kind 0 artifact\n.entry 0\n.function main 0 0 0 int 1\nARR_NEW 1\nPUSH_I64 7\nARR_PUSH\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n'
                for case,text,refusal in (("copy",positive,None),("cross-image",negative,"own image"),("input-alias",alias,"independent")):
                    source,module = folder/(case+".nasm"),folder/(case+".nvm");source.write_text(text)
                    module_hash = None
                    for stage,argv in (("assemble",[assembler,"asm",str(source),"-o",str(module)]),("verify",[vm,"--verify-only",str(module)]),("execute",[vm,str(module)])):
                        result=subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=60)
                        label="cli-"+case+"-"+stage
                        (work/(label+".stdout")).write_bytes(result.stdout);(work/(label+".stderr")).write_bytes(result.stderr)
                        operations.append({"label":label,"argv":argv,"returncode":result.returncode})
                        (work/"operations.json").write_text(json.dumps(operations,indent=2)+"\n")
                        if stage=="execute" and refusal:
                            self.assertNotEqual(result.returncode,0)
                            self.assertIn(refusal,result.stderr.decode(errors="replace"))
                        else:self.assertEqual(result.returncode,0,result.stderr.decode(errors="replace"))
                        if stage=="assemble":module_hash=hashlib.sha256(module.read_bytes()).hexdigest()
                        else:self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),module_hash)
                self.assertEqual(tools,{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in tools})
                (work/"cli-tools-before-after.json").write_text(json.dumps(tools,indent=2)+"\n")
            self.assertEqual(identities,{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in identities})
            products.append({"mode":name,"sha256_before_after":identities,
                             "scope":"I execute one native driver and load/call three native shared libraries."})
            (work/"products.json").write_text(json.dumps(products,indent=2)+"\n")
        self.assertEqual(hashlib.sha256(launcher.read_bytes()).hexdigest(),launcher_hash)
        (work/"compiler.json").write_text(json.dumps({"argv":cc,"observed_launcher":str(launcher),"sha256_before_after":launcher_hash,"version":version,"scope":"I record the selected compiler launcher, not complete SDK/toolchain closure."},indent=2)+"\n")
        (work/"scope.json").write_text(json.dumps({"vm_cli_requested":bool(os.environ.get("NANO_SNAPSHOT_VM")),"scope":"I execute VM CLI controls only with the explicitly selected fresh VM and assembler. Otherwise I qualify native loader controls alone."},indent=2)+"\n")
        self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
        (work/"inputs-before-after.json").write_text(json.dumps(before,indent=2)+"\n")


if __name__ == "__main__":
    unittest.main()
