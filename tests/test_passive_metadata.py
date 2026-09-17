"""I retain validated graph claims while serial VM and AOT execute the same code."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
class PassiveMetadata(unittest.TestCase):
    def test_codec_validation_and_serial_execution(self):
        with tempfile.TemporaryDirectory(prefix='nano-passive-') as tmp:
            module = Path(tmp) / 'graph.nvm'
            generated = Path(tmp) / 'graph.c'
            native = Path(tmp) / 'graph'
            commands = [([ROOT/'obj/test_passive',module],None),
                        ([ROOT/'bin/nano_vm',module],b'42\n'),
                        ([ROOT/'bin/nvm2c',module,'-o',generated],None),
                        (['cc','-std=c11','-Wall','-Wextra','-Werror',generated,'-lm','-o',native],None),
                        ([native],b'42\n')]
            for index in range(3):
                branch = Path(str(module) + f'.branch{index}')
                commands.extend([([ROOT/'bin/nano_vm', branch], b'42\n'),
                                 ([ROOT/'bin/nvm2c', branch, '-o', generated], None),
                                 (['cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                                   generated, '-lm', '-o', native], None),
                                 ([native], b'42\n')])
            for command, output in commands:
                result=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=90)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                if output is not None: self.assertEqual(result.stdout,output)
if __name__=='__main__': unittest.main()
