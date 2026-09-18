"""I preserve the existing source escape decoder and visible C-string bytes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
CASES=[(r'alpha\nbeta',b'alpha\nbeta'),(r'\t8',b'\t8'),(r'\r7',b'\r7'),
       (r'\"',b'"'),(r"\'",b"'"),(r'\\',b'\\'),
       (r'\q\x41\u0041\123',b'\\q\\x41\\u0041\\123'),
       (r'head\0tail',b'head'),(r'\0ignored',b''),('',b''),
       ('café λ','café λ'.encode()),('??/??=',b'??/??='),('\x017',b'\x017')]
SOURCE='fn main()->int{\n'+''.join('(print "'+raw+'") (print "|")\n' for raw,_ in CASES)+r'''assert (== "row\nend" "row
end")
assert (== "keep\0discard" "keep")
return 0}
shadow main{assert true}
'''
EXPECTED=b''.join(expected+b'|' for _,expected in CASES)
class PublicCLiterals(unittest.TestCase):
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-public-c-literal-'))
        print('I retain fresh literal evidence at',self.work,flush=True)
    def run_cmd(self,args):
        result=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,timeout=240)
        self.assertEqual(result.returncode,0,f'command={args!r}\n{result.stdout!r}\n{result.stderr!r}')
        self.assertNotIn(b'Sanitizer',result.stderr);self.assertNotIn(b'runtime error:',result.stderr)
        return result
    def test_exact_visible_bytes_all_source_routes(self):
        source=self.work/'ordinary.nano';source.write_text(SOURCE)
        output=self.work/'ordinary.c';self.run_cmd([ROOT/'bin/nanoc_c','--target','c',source,'-o',output])
        text=output.read_text();self.assertNotIn('({',text)
        self.assertIn(r'\0017',text);self.assertIn(r'\?\?/',text);self.assertIn(r'\303\251',text)
        for standard in ('c99','c11'):
            for opt in ('-O0','-O2'):
                exe=self.work/(standard+opt)
                self.run_cmd([os.environ.get('CC','cc'),'-std='+standard,'-pedantic-errors',
                    '-Werror=implicit-function-declaration','-Werror=return-type',opt,
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all',output,'-lm','-o',exe])
                self.assertEqual(self.run_cmd([exe]).stdout,EXPECTED)
        self.assertEqual(self.run_cmd([ROOT/'bin/nano',source]).stdout,EXPECTED)
        module=self.work/'ordinary.nvm';self.run_cmd([ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module])
        self.run_cmd([ROOT/'bin/nano_vm','--verify-only',module]);self.assertEqual(self.run_cmd([ROOT/'bin/nano_vm',module]).stdout,EXPECTED)
        native=self.work/'native.c';self.run_cmd([ROOT/'bin/nvm2c',module,'-o',native])
        exe=self.work/'native';self.run_cmd([os.environ.get('CC','cc'),'-std=c11','-O2','-fsanitize=address,undefined','-fno-sanitize-recover=all',native,'-lm','-o',exe]);self.assertEqual(self.run_cmd([exe]).stdout,EXPECTED)
    def test_decoder_allocation_refusal_and_publication_recovery(self):
        exe=self.work/'api';self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_literal_api.c','-o',exe]);self.run_cmd([exe,self.work/'previous.c'])
if __name__=='__main__':unittest.main()
