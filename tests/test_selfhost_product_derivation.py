"""I check that my self-hosted compiler derives C only from the verified NanoISA module (row 10684)."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_MODULE', ROOT / 'bin/nanoc_stage1.nvm'))
VM = ROOT / 'bin/nano_vm'
NVM2C = ROOT / 'bin/nvm2c'


class SelfHostProductDerivation(unittest.TestCase):
    def run_tool(self, command):
        result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, ' '.join(map(str, command)) + '\n' + result.stdout + result.stderr)

    def test_target_c_is_nvm2c_of_the_emitted_module(self):
        missing = [str(path) for path in (COMPILER, VM, NVM2C) if not path.exists()]
        if missing:
            self.skipTest('I require built tools and a bootstrapped compiler: ' + ', '.join(missing))
        with tempfile.TemporaryDirectory(prefix='nano-product-derivation-') as directory:
            work = Path(directory)
            programs = {'hello': 'fn main() -> int { (println "hi") return 0 }\nshadow main { assert true }\n',
                        'other': 'fn main() -> int { (println "different") return 0 }\nshadow main { assert true }\n'}
            derived = {}
            for name, text in programs.items():
                source = work / (name + '.nano')
                source.write_text(text)
                module, direct, via_module = work / (name + '.nvm'), work / (name + '.c'), work / (name + '.via.c')
                self.run_tool([VM, COMPILER, '--', source, '--emit-nvm', '-o', module])
                self.run_tool([VM, COMPILER, '--', source, '--target', 'c', '-o', direct])
                self.run_tool([NVM2C, module, '-o', via_module])
                derived[name] = (direct.read_bytes(), via_module.read_bytes())
                self.assertEqual(derived[name][0], derived[name][1],
                                 'I found C that does not come from the verified module for ' + name)
                self.assertGreater(len(derived[name][0]), 0)
            # I require the comparison to distinguish programs, so identical empty outputs cannot pass.
            self.assertNotEqual(derived['hello'][0], derived['other'][0])


if __name__ == '__main__':
    unittest.main()
