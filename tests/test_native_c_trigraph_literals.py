"""I preserve ordinary literal bytes across C preprocessing."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
def quoted(value):
    return '"'+value.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\t','\\t').replace('\r','\\r')+'"'
class NativeLiteralBytes(unittest.TestCase):
    def checked(self,args):
        r=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,timeout=120)
        self.assertEqual(r.returncode,0,f'command={args!r}\n{r.stdout!r}\n{r.stderr!r}')
        self.assertNotIn(b'Sanitizer',r.stderr);self.assertNotIn(b'runtime error:',r.stderr)
        return r
    def paired(self,values,cast_only=False):
        work=Path(tempfile.mkdtemp(prefix='nano-native-literal-bytes-'))
        print('I retain ordinary literal evidence at',work,flush=True)
        text=''.join(f'.string s{i} {quoted(value)}\n' for i,value in enumerate(values))
        text+='.entry main\n.function main 0 0 0 int 1\n'
        for i,value in enumerate(values):
            text+=f'PUSH_STR s{i}\nDUP\nSTR_LEN\nPUSH_I64 {len(value.encode())}\nEQ\nASSERT\nPRINTLN\n'
        if cast_only: text+='PUSH_F64 1.5\nCAST_STRING\nSTR_LEN\nPUSH_I64 3\nEQ\nASSERT\n'
        text+='PUSH_I64 0\nRET\n.end\n'
        assembly=work/'ordinary.nasm';assembly.write_text(text)
        module=work/'ordinary.nvm';self.checked([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
        self.checked([ROOT/'bin/nano_vm','--verify-only',module])
        expected=''.join(value+'\n' for value in values).encode()
        self.assertEqual(self.checked([ROOT/'bin/nano_vm',module]).stdout,expected)
        source=work/'ordinary.c';self.checked([ROOT/'bin/nvm2c',module,'-o',source])
        self.assertNotIn('??',source.read_text())
        for opt in ('-O0','-O2'):
            exe=work/opt;self.checked([os.environ.get('CC','cc'),'-std=c11',opt,'-Wall','-Wextra','-Werror',
                                      '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-lm','-o',exe])
            self.assertEqual(self.checked([exe]).stdout,expected)
    def test_all_nine_trigraph_like_sequences(self):
        self.paired(['left??'+last+'right' for last in "=/\'()!<>-"])
    def test_cast_only_shared_provider(self):
        self.paired([],cast_only=True)
    def test_neighbor_literal_bytes_and_lengths(self):
        self.paired(['','?','???','edge?\\"8','café λ','\x017','row\nnext','\t8\r7'])
if __name__=='__main__':unittest.main()
