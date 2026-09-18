"""I validate runtime packaging independently of executable opcode admission."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from scripts.embed_managed_runtime import generate


class ManagedRuntimePackage(unittest.TestCase):
    def run_cmd(self, args):
        result = subprocess.run([str(x) for x in args], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, str(args)+'\n'+result.stdout+result.stderr)
        return result.stdout

    def test_reproducible_target_packages_and_real_links(self):
        clang = shlex.split(os.environ.get('NMS_RUNTIME_CLANG', 'clang'))
        header, manifest, variants = generate(clang, ['opt'])
        again = generate(clang, ['opt'])
        self.assertEqual((header, manifest), again[:2])
        self.assertNotEqual(variants['native']['triple'], variants['wasm32']['triple'])
        with tempfile.TemporaryDirectory(prefix='nano-runtime-package-') as directory:
            work = Path(directory)
            (work/'runtime.h').write_text(header)
            # I exercise the exact C-embedded bytes, not only the generator's strings.
            (work/'dump.c').write_text('#include <stdio.h>\n#include "runtime.h"\n'
                'int main(int argc,char **argv) {(void)argv; return fputs(argc>1 ? '
                'nms_runtime_ir_wasm32 : nms_runtime_ir_native,stdout)<0;}\n')
            self.run_cmd(shlex.split(os.environ.get('CC','cc'))+[work/'dump.c','-o',work/'dump'])
            for target, variant in variants.items():
                dumped = self.run_cmd([work/'dump']+(['wasm'] if target == 'wasm32' else []))
                self.assertEqual(dumped, variant['ir'])
                # A normal application definition follows the complete runtime module.
                # Calls cross its scalar ABI; no private C aggregate layout is guessed.
                application = '\n@package_name = private constant [15 x i8] c"nano_try_entry\\00"\n' \
                    'define i32 @main() {\n %v = call i32 @nms_reserved_entry(ptr @package_name)\n' \
                    ' %result = sub i32 1, %v\n ret i32 %result\n}\n'
                ir = work/(target+'.ll')
                ir.write_text(dumped+application.replace('@main()', '@nano_package()') if target == 'wasm32' else dumped+application)
                self.run_cmd(['opt','-passes=verify','-disable-output',ir])
                if target == 'native':
                    self.run_cmd(clang+shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS',''))+[ir,'-o',work/'native'])
                    self.run_cmd([work/'native'])
                else:
                    wasm = work/'runtime.wasm'
                    self.run_cmd(clang+['--target=wasm32-unknown-unknown','-nostdlib',ir,
                        '-Wl,--no-entry','-Wl,--export=nano_package','-Wl,--fatal-warnings','-o',wasm])
                    self.assertEqual(self.run_cmd(['wasmtime','run','--invoke','nano_package',wasm]), '0\n')
                    script = "const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');const e=new WebAssembly.Instance(m).exports;if(e.nano_package())throw Error('runtime ABI');"
                    self.run_cmd(['node','-e',script,wasm])
            self.assertEqual(set(manifest['sources']), {'src/nanoisa/managed_strings.c',
                'src/nanoisa/managed_strings.h', 'scripts/embed_managed_runtime.py'})
            self.assertEqual(json.loads(json.dumps(manifest)),manifest)


if __name__ == '__main__':
    unittest.main()
