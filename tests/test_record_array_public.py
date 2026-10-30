"""I exercise shared array-field authority through actual public consumers."""
import os
import re
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import unittest
from tests.test_record_array_vm import PROVIDERS as VM_PROVIDERS

ROOT = Path(__file__).resolve().parents[1]


class RecordArrayPublic(unittest.TestCase):
    def test_public_vm_bridge_and_sanitized_native(self):
        artifacts = Path(tempfile.mkdtemp(prefix='nano-record-array-public-'))
        print(f'I retain public-route evidence at {artifacts}', flush=True)
        def run(name, command, build=False, expected=0):
            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                    timeout=int(os.environ.get('CMD_TIMEOUT', '1800')) if build else 120)
            (artifacts / (name + '.stdout')).write_bytes(result.stdout)
            (artifacts / (name + '.stderr')).write_bytes(result.stderr)
            self.assertEqual(result.returncode, expected, result.stdout.decode(errors='replace') + result.stderr.decode(errors='replace'))
            return result
        # I instrument the actual public VM and proof/transport providers,
        # keeping the ordinary ABI (no private-runtime compile definition).
        cc = shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_CC', 'cc'))
        providers = list(dict.fromkeys([*VM_PROVIDERS, 'src/nanoisa/nvm_v2_convert.c',
            'src/nanoisa/nvm_v2_module.c', 'src/nanoisa/nvm_format.c', 'src/nanoisa/nvm2c.c']))
        excluded = {Path(source).stem for source in providers}
        common = [obj for obj in shlex.split(os.environ['RECORD_ARRAY_PUBLIC_OBJECTS'])
                  if Path(obj).stem not in excluded]
        # I preserve single-pass Unix archive resolution after object users.
        common = [p for p in common if Path(p).suffix != '.a'] + [p for p in common if Path(p).suffix == '.a']
        flags = [*shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_CFLAGS', '')),
                 '-std=c11', '-D_DEFAULT_SOURCE', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                 '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-Isrc', '-Isrc/nanoisa']
        objects = []
        for source in providers:
            obj = artifacts / (Path(source).stem + '.o')
            run(obj.stem + '-instrument', [*cc, *flags, '-c', source, '-o', str(obj)], build=True)
            objects.append(str(obj))
        instrumented = artifacts / 'public-api-instrumented'
        run('instrumented-link', [*cc, *flags, 'tests/nanoisa/test_record_array_public.c',
            'tests/nanoisa/record_array_alloc.c', *objects, *common,
            *shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_LDFLAGS', '-lm')),
            '-o', str(instrumented)], build=True)
        run('public-api-instrumented', [str(instrumented), str(artifacts)])
        observed = []
        for source in VM_PROVIDERS:
            obj = artifacts / ('observed-' + Path(source).stem + '.o')
            run(obj.stem + '-build', [*cc, *flags, '-DRA_ALLOC_WRAP',
                '-include', 'tests/nanoisa/record_array_alloc.h', '-c', source,
                '-o', str(obj)], build=True)
            observed.append(str(obj))
        observed_names = {Path(source).stem for source in VM_PROVIDERS}
        observed_common = [obj for obj in shlex.split(os.environ['RECORD_ARRAY_PUBLIC_OBJECTS'])
                           if Path(obj).stem not in observed_names]
        observed_common = [p for p in observed_common if Path(p).suffix != '.a'] + [p for p in observed_common if Path(p).suffix == '.a']
        observed_exe = artifacts / 'public-api-allocation-prefixes'
        run('observed-link', [*cc, *flags, '-DRA_PUBLIC_ALLOCATIONS',
            'tests/nanoisa/test_record_array_public.c', 'tests/nanoisa/record_array_alloc.c',
            *observed, *observed_common,
            *shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_LDFLAGS', '-lm')),
            '-o', str(observed_exe)], build=True)
        observed_stdout=b''
        for fixture in ('flow','mutation'):
            for tag in range(1,6):
                observed=run(f'public-allocation-{fixture}-{tag}',
                    [str(observed_exe),str(artifacts),fixture,str(tag)])
                observed_stdout+=observed.stdout
        (artifacts/'public-allocation-prefixes.stdout').write_bytes(observed_stdout)
        self.assertEqual(observed_stdout.count(b'transient and persistent public allocation prefixes'),5)
        self.assertEqual(observed_stdout.count(b'transient and persistent public mutation allocation prefixes'),5)
        receipt = run('public-api', [str(ROOT / 'bin/test_record_array_public'), str(artifacts)])
        self.assertIn(b'public record-array checks', receipt.stdout)
        sources = sorted([*artifacts.glob('array-*.c'), *artifacts.glob('flow-*.c'),
                          *artifacts.glob('mutation-*.c')])
        self.assertEqual(len(sources), 15)
        sources.append(artifacts/'branch-uninitialized.c')
        failures=sorted(artifacts.glob('failure-mutation-*.c'))
        self.assertEqual(len(failures),15)
        sources.extend(failures)
        cc = shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_CC', 'cc'))
        for source in sources:
            for optimization in ('-O0', '-O2'):
                name = source.stem + optimization[1:]
                executable = artifacts / name
                run(name + '-build', [*cc, '-std=c11', optimization, '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', str(source),
                    str(ROOT / 'bin/nano_aot_runtime.o'), '-lm',
                    *shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_LDFLAGS', '')),
                    '-o', str(executable)], build=True)
                failed=source in failures
                result=run(name + '-execute', [str(executable)], expected=-signal.SIGABRT if failed else 0)
                if failed:
                    generated=source.read_text()
                    assertions=list(re.finditer(r't\[(\d+)\] = 0LL;\n\s*if \(!t\[\1\]\) NVM2C_ABORT\(\);',generated))
                    self.assertEqual(len(assertions),1)
                    line=generated.count('\n',0,assertions[0].end())+1
                    self.assertIn(f'I stopped at a native invariant in nl_main at generated C line {line}.\n'.encode(),result.stderr)
                    self.assertNotIn(b'ERROR: AddressSanitizer',result.stderr)
                    self.assertNotIn(b'runtime error:',result.stderr)

        identity_source=artifacts/'canonical-identity.c'
        identity_source.write_text('#include <stdlib.h>\n#include <setjmp.h>\n'
            'static int fail_record_allocation, fault_armed; static jmp_buf record_fault;\n'
            'static void *record_calloc(size_t n,size_t z) { if(fail_record_allocation) return NULL; return calloc(n,z); }\n'
            '_Noreturn static void record_abort(void) { if(!fault_armed) exit(97); longjmp(record_fault,1); }\n'
            '#define calloc record_calloc\n#define abort record_abort\n#define main fixture_main\n'+
            (artifacts/'array-1.c').read_text()+ '\n'+
            (ROOT/'tests/native_record_identity_probe.c').read_text())
        for optimization in ('-O0','-O2'):
            name='canonical-identity'+optimization[1:];executable=artifacts/name
            run(name+'-build',[*cc,'-std=c11',optimization,'-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(identity_source),
                str(ROOT/'bin/nano_aot_runtime.o'),'-lm',
                *shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_LDFLAGS','')),
                '-o',str(executable)],build=True)
            run(name+'-execute',[str(executable)])


if __name__ == '__main__':
    unittest.main()
