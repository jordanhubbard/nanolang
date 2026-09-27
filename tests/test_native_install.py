"""I use the actual installed native package outside the repository."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeInstall(unittest.TestCase):
    def test_installed_scalar_product_and_uninstall(self):
        with tempfile.TemporaryDirectory(prefix='nano-native-install-') as tmp:
            work = Path(tmp)
            prefix = work/"prefix with ' quote"
            result = subprocess.run(['make','install','PREFIX='+str(prefix)],cwd=ROOT,
                                    capture_output=True,text=True,timeout=900)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue((prefix/'bin/nano_aot_runtime.o').is_file())
            module = work/"foreign module's source"
            module.mkdir()
            (module/'module.json').write_text(json.dumps({'name':'installed_scalar','c_sources':['provider.c']}))
            (module/'provider.c').write_text('#include <stdint.h>\nint64_t installed_answer(void) { return 37; }\n')
            (module/'api.nano').write_text('module InstalledScalar\nextern fn installed_answer() -> int\n'
                'pub fn answer() -> int { unsafe { return (installed_answer) } }\n'
                'shadow answer { assert (== (answer) 37) }\n')
            source = work/'main.nano'
            source.write_text('module '+json.dumps(str(module/'api.nano'))+' as fixture\n'
                'fn main() -> int { assert (== (fixture.answer) 37) return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            env = os.environ.copy()
            for key in ('NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH',
                        'NANO_AS_CAPTURE_HELPER','NANOLANG_SELFHOST_COMPILER'):
                env.pop(key,None)
            env['PATH'] = str(prefix/'bin')+os.pathsep+env['PATH']
            env['NANO_BUILD_CACHE'] = str(work/'cache')
            alias = work/'compiler-link'
            alias.symlink_to(prefix/'bin/nanoc')
            invocations = [str(prefix/'bin/nanoc'), 'nanoc', os.path.relpath(prefix/'bin/nanoc',work), str(alias)]
            for index, compiler in enumerate(invocations):
                with self.subTest(invocation=compiler):
                    output = work/('program-'+str(index))
                    result = subprocess.run([compiler,source,'-o',output],cwd=work,env=env,
                                            capture_output=True,text=True,timeout=120)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    result = subprocess.run([output],cwd=work,env=env,capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            # I honor explicit overrides even when a working sibling is installed.
            output = work/'preserved'
            output.write_text('prior output')
            result = subprocess.run([prefix/'bin/nanoc',source,'-o',output],cwd=work,
                env={**env,'NANO_NVM2C':str(work/'missing-translator')},
                capture_output=True,text=True,timeout=120)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(output.read_text(),'prior output')
            result = subprocess.run(['make','uninstall','PREFIX='+str(prefix)],cwd=ROOT,
                                    capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for path in ('bin/nano_aot_runtime.o','bin/nanoc','bin/nvm2c','bin/nano_vm','lib/libnano_file_runtime.a'):
                self.assertFalse((prefix/path).exists(),path)
