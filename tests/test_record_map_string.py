"""Selected STRING map values: ownership, observation and native failure boundaries."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import unittest
from unittest.mock import patch
from tests import test_record_map_runtime as runtime


class RecordMapString(runtime.RecordMapRuntime):
    harness_source = 'tests/nanoisa/test_record_map_string.c'
    result_banner = b'selected STRING-value consumer assertions'

    def command(self, name, args, **kwargs):
        if name == 'runtime-run':
            with patch.dict(os.environ, {'MAP_STRING_NATIVE_DIR': str(self.artifacts)}):
                return super().command(name, args, **kwargs)
        return super().command(name, args, **kwargs)

    def runtime(self, computed_goto):
        with patch.dict(os.environ, {'RECORD_MAP_MODULES': os.environ.get('RECORD_MAP_STRING_MODULES', '')}):
            super().runtime(computed_goto)
        if not computed_goto:
            self.native()

    def native(self):
        cc = shlex.split(os.environ.get('RECORD_ARRAY_VM_CC', 'cc'))
        rows = []

        def run(name, args, failure=None):
            result = subprocess.run(list(map(str, args)), capture_output=True, timeout=180)
            (self.artifacts/(name+'.stdout')).write_bytes(result.stdout)
            (self.artifacts/(name+'.stderr')).write_bytes(result.stderr)
            rows.append({'name': name, 'argv': list(map(str, args)), 'status': result.returncode})
            (self.artifacts/'string-native-commands.json').write_text(json.dumps(rows, indent=2)+'\n')
            self.assertNotIn(b'AddressSanitizer', result.stderr)
            self.assertNotIn(b'runtime error:', result.stderr)
            if failure:
                self.assertEqual(result.returncode, -signal.SIGABRT)
                function = 'nvalue_require_string' if failure == 'string' else '[A-Za-z0-9_]+'
                self.assertRegex(result.stderr, ('^I stopped at a native invariant in '+function+r' at generated C line [0-9]+\.\n$').encode())
                self.assertEqual(result.stdout, b'')
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, b'')
            return result.stdout

        def compile_native(name, text, opt):
            source = self.artifacts/(name+'.c'); source.write_text(text)
            exe = self.artifacts/name
            run(name+'-build', [*cc, '-std=c11', '-g', '-O'+opt, '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all', source, '-lm', '-o', exe])
            return exe

        needle = 'if (debt >= nmap_collection_budget) nmap_collect();'
        for variant in range(3):
            text = (self.artifacts/f'string-{variant}.c').read_text()
            self.assertEqual(text.count(needle), 1)
            forced = ('#define main fixture_main\n'+text.replace(needle, '(void)debt; nmap_collect();')+
                '\n#undef main\nint main(void){int r=fixture_main();'
                'for(size_t i=0;i<sizeof nglobal/sizeof nglobal[0];i++)nglobal[i]=(nmap_value){0};'
                'nmap_collect();if(nmap_owned_head||nrec_owned_head||nmap_live_bytes||nagg_live_bytes||nroot_head'+
                ('||nstr_live_bytes' if 'nstr_live_bytes' in text else '')+')abort();return r;}\n')
            for mode, body in [('raw', text), ('forced', forced)]:
                for opt in ('0', '2'):
                    name = f'string-{variant}-{mode}-O{opt}'
                    exe = compile_native(name, body, opt)
                    self.assertEqual(run(name+'-run', [exe], 'string' if variant == 2 else None), b'')
        text = (self.artifacts/'string-0.c').read_text()
        probe = Path('tests/nanoisa/native_map_alloc_probe.h').read_text()
        body = (probe+'\n#define main fixture_main\n'+text.replace(needle, '(void)debt; nmap_collect();')+
            '\n#undef main\n#undef malloc\n#undef calloc\n#undef realloc\n#undef free\n'
            'int main(int argc,char **argv){if(argc>1)ntest_fail=(size_t)strtoull(argv[1],NULL,10);'
            'if(argc>2)ntest_persistent=1;int r=fixture_main();if(ntest_live||ntest_bytes)abort();'
            'printf("%zu %zu %zu\\n",ntest_calls,ntest_live,ntest_bytes);return r;}\n')
        counts = {}
        for opt in ('0', '2'):
            name = 'string-allocation-O'+opt; exe = compile_native(name, body, opt)
            measured = run(name+'-measure', [exe]); count, live, size = map(int, measured.split())
            self.assertGreater(count, 0); self.assertEqual((live, size), (0, 0)); counts[name] = count
            for persistent in (False, True):
                for position in range(count):
                    run(f'{name}-denied-{persistent}-{position}', [exe, position]+(['persistent'] if persistent else []), 'allocation')
            self.assertEqual(run(name+'-retry', [exe]), measured)
        (self.artifacts/'string-native-allocation-counts.json').write_text(json.dumps(counts, indent=2)+'\n')
        print('I checked all generated STRING-value allocation positions in both modes:', counts, flush=True)


if __name__ == '__main__':
    unittest.main()
