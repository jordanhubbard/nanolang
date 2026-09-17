"""I root owned builtin host results and copies of borrowed facade strings."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeHostStrings(unittest.TestCase):
    def command(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def build_run(self, work, text, args=(), env=None, vm=True):
        asm, module, source, binary = [work / x for x in ('input.nasm', 'input.nvm', 'input.c', 'program')]
        asm.write_text(text)
        self.command([ROOT / 'bin/nanoisa', 'asm', asm, '-o', module])
        environment = {**os.environ, **(env or {})}
        if vm:
            self.command([ROOT / 'bin/nano_vm', module, '--', *args], env=environment)
        self.command([ROOT / 'bin/nvm2c', module, '-o', source])
        generated = source.read_text()
        self.assertIn('nstr_release_owned();', generated)
        generated = generated.replace('    return result;\n}',
            '    if(nstr_live_bytes || nstr_owners || nstr_peak_bytes > 70000) abort();\n'
            '    return result;\n}')
        source.write_text(generated)
        self.command(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                      source, '-o', binary, '-lm', '-ldl'])
        return self.command([binary, *args], env={**environment, 'ASAN_OPTIONS': 'detect_leaks=1'})

    def test_argv_environment_copies_without_allocating_string_opcodes(self):
        text = ('.import "" "get_argv" string int\n.import "" "vm_getenv" string string\n'
                '.import "" "vm_tmp_dir" string\n.import "" "vm_getcwd" string\n'
                '.string expected "5"\n.string env "NANO_HOST_COPY_TEST"\n.string value "borrowed-env"\n'
                '.entry main\n.function main 0 2 0 int 1\nCALL relay\nSTORE_GLOBAL 0\n'
                'PUSH_I64 -1\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n'
                'PUSH_STR env\nCALL_EXTERN 1\nSTORE_LOCAL 1\nPUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
                'PUSH_I64 1\nCALL_EXTERN 0\nPOP\nPUSH_STR env\nCALL_EXTERN 1\nPOP\n'
                'CALL_EXTERN 2\nPOP\nCALL_EXTERN 3\nPOP\n'
                'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nPUSH_I64 5000\nI64_LT_S\nJMP_TRUE loop\n'
                'LOAD_GLOBAL 0\nPUSH_STR expected\nEQ\nASSERT\n'
                'LOAD_LOCAL 1\nPUSH_STR value\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
                '.function make 0 0 0 string 1\nPUSH_I64 1\nCALL_EXTERN 0\nRET\n.end\n'
                '.function relay 0 0 0 string 1\nTAIL_CALL make\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-host-argv-') as tmp:
            self.build_run(Path(tmp), text, args=('5',), env={'NANO_HOST_COPY_TEST': 'borrowed-env'})

    def test_builtin_file_capture_and_temp_results(self):
        with tempfile.TemporaryDirectory(prefix='nano-host-files-') as tmp:
            work=Path(tmp); (work/'text').write_text('retained'); (work/'empty').write_text(''); (work/'nul').write_bytes(b'a\0b')
            strings=''.join(f'.string {name} {json.dumps(str(work/name))}\n' for name in ('text','empty','nul','missing'))
            text=('.import "" "vm_file_read" string string\n'
                  '.import "" "nl_exec_capture" string string\n.import "" "vm_mktemp_dir" string string\n'+strings+
                  '.string value "retained"\n.string path "a/./b/../c"\n.string normal "a/c"\n'
                  '.string command "printf retained"\n.string prefix "owned_"\n.entry main\n'
                  '.function main 0 2 0 int 1\nPUSH_STR text\nCALL_EXTERN 0\nSTORE_LOCAL 1\n'
                  'PUSH_I64 0\nSTORE_LOCAL 0\nloop:\nPUSH_STR text\nCALL_EXTERN 0\nPOP\n'
                  'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 3000\nI64_LT_S\nJMP_TRUE loop\n'+
                  ''.join(f'PUSH_STR {name}\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n' for name in ('empty','nul','missing'))+
                  'PUSH_STR command\nCALL_EXTERN 1\nPUSH_STR value\nEQ\nASSERT\n'
                  'PUSH_STR prefix\nCALL_EXTERN 2\nSTR_LEN\nPUSH_I64 0\nI64_GT_S\nASSERT\n'
                  'LOAD_LOCAL 1\nPUSH_STR value\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
            self.build_run(work,text,env={'TMPDIR':str(work)})

    def test_builtin_normalize_churn(self):
        text=('.import "" "path_normalize" string string\n.string path "a/./b/../c"\n'
              '.string normal "a/c"\n.entry main\n.function main 0 1 0 int 1\n'
              'PUSH_I64 0\nSTORE_LOCAL 0\nloop:\nPUSH_STR path\nCALL_EXTERN 0\n'
              'PUSH_STR normal\nEQ\nASSERT\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
              'LOAD_LOCAL 0\nPUSH_I64 10000\nI64_LT_S\nJMP_TRUE loop\nPUSH_I64 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-host-normalize-') as tmp:
            for alias in ('path_normalize', 'nl_os_path_normalize'):
                with self.subTest(alias=alias):
                    self.build_run(Path(tmp),text.replace('"path_normalize"', '"'+alias+'"'))

    def test_snapshot_copies_borrowed_facade_and_preserves_literal_artifact(self):
        with tempfile.TemporaryDirectory(prefix='nano-host-snapshot-') as tmp:
            work=Path(tmp); library=work/'host.so'; host=work/'host.c'
            host.write_text('#include <stdio.h>\nconst char *nl_nanoisa_last_error(void) { static char text[32]; static int count; snprintf(text,sizeof text,"snapshot-%d",++count); return text; }\nconst char *path_basename(const char *s) { (void)s; return "borrowed-literal"; }\nconst char *path_normalize(const char *s) { (void)s; return "borrowed-literal"; }\n')
            self.command(['cc','-shared','-fPIC',host,'-o',library])
            text=(f'.import {json.dumps(str(library))} "nl_nanoisa_last_error" string\n.import_kind 0 artifact\n'
                  f'.import {json.dumps(str(library))} "path_basename" string string\n.import_kind 1 artifact\n'
                  f'.import {json.dumps(str(library))} "path_normalize" string string\n.import_kind 2 artifact\n'
                  '.string first "snapshot-1"\n.string literal "borrowed-literal"\n.entry main\n'
                  '.function main 0 2 0 int 1\nCALL_EXTERN 0\nSTORE_LOCAL 1\nPUSH_I64 0\nSTORE_LOCAL 0\n'
                  'loop:\nCALL_EXTERN 0\nPOP\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
                  'LOAD_LOCAL 0\nPUSH_I64 5000\nI64_LT_S\nJMP_TRUE loop\n'
                  'LOAD_LOCAL 1\nPUSH_STR first\nEQ\nASSERT\nPUSH_STR first\nCALL_EXTERN 1\nPUSH_STR literal\nEQ\nASSERT\nPUSH_STR first\nCALL_EXTERN 2\nPUSH_STR literal\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
            self.build_run(work,text)


if __name__ == '__main__':
    unittest.main()
