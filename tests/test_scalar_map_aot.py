"""I retain exact map pairs and detached enumeration through my VM and C AOT."""
from pathlib import Path
import json
import os
import shlex
import signal
import time
import subprocess
import tempfile
import unittest
from tests.test_native_optional_array_reads import OptionalArrayReads

ROOT = Path(__file__).resolve().parents[1]
HEADER = '.entry main\n.string first "first"\n.string second "second"\n.string kept "retained"\n.string changed "changed"\n'
CHURN = ('.function churn 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nagain:\n'
         'HM_NEW 5 5\nPUSH_STR first\nPUSH_STR kept\nHM_SET\nPOP\n'
         'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
         'LOAD_LOCAL 0\nPUSH_I64 5000\nI64_LT_S\nJMP_FALSE done\nJMP again\n'
         'done:\nRET\n.end\n')


class ScalarMapAot(unittest.TestCase):
    checked = OptionalArrayReads.checked
    paired = OptionalArrayReads.paired

    def test_all_pairs_empty_growth_optional_writes_and_detached_snapshots(self):
        for key in (1, 5):
            for value in (1, 5):
                with self.subTest(key=key, value=value):
                    push_key = 'PUSH_I64 -9223372036854775808\n' if key == 1 else 'PUSH_STR first\n'
                    push_value = 'PUSH_I64 -7\n' if value == 1 else 'PUSH_STR kept\n'
                    changed = 'PUSH_I64 19\n' if value == 1 else 'PUSH_STR changed\n'
                    body = f'HM_NEW {key} {value}\nSTORE_LOCAL 0\n'
                    for opcode, tag, push in [('HM_KEYS', key, push_key), ('HM_VALUES', value, push_value)]:
                        body += 'LOAD_LOCAL 0\n' + opcode + '\nDUP\nARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n'
                        body += push + 'ARR_PUSH\nPUSH_I64 0\nARR_GET\nDUP\n'
                        body += f'TYPE_CHECK {tag}\nASSERT\n' + push + 'EQ\nASSERT\n'
                    body += 'LOAD_LOCAL 0\n' + push_key + push_value + 'HM_SET\nPOP\n'
                    body += 'LOAD_LOCAL 0\nHM_KEYS\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nHM_VALUES\nSTORE_LOCAL 2\n'
                    body += 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nLOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\nHM_SET\nPOP\n'
                    body += 'LOAD_LOCAL 0\n' + push_key + changed + 'HM_SET\nPOP\n'
                    body += 'LOAD_LOCAL 0\n' + push_key + 'HM_GET\n' + changed + 'EQ\nASSERT\n'
                    body += 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nHM_DELETE\nPOP\n'
                    body += 'LOAD_LOCAL 0\nHM_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n'
                    body += f'HM_NEW {key} {value}\nSTORE_LOCAL 0\nCALL churn\n'
                    for slot, tag, push in [(1, key, push_key), (2, value, push_value)]:
                        body += f'LOAD_LOCAL {slot}\nDUP\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\n'
                        body += f'PUSH_I64 0\nARR_GET\nDUP\nTYPE_CHECK {tag}\nASSERT\n' + push + 'EQ\nASSERT\n'
                    self.paired(HEADER + '.function main 0 3 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n' + CHURN)

    def test_forward_map_parameters_and_returned_snapshot_arrays(self):
        for key in (1, 5):
            for value in (1, 5):
                with self.subTest(key=key, value=value):
                    push_key = 'PUSH_I64 9223372036854775807\n' if key == 1 else 'PUSH_STR first\n'
                    push_value = 'PUSH_I64 -7\n' if value == 1 else 'PUSH_STR kept\n'
                    body = f'HM_NEW {key} {value}\n' + push_key + push_value + 'HM_SET\nSTORE_LOCAL 0\n'
                    body += 'LOAD_LOCAL 0\nCALL keys\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nCALL values\nSTORE_LOCAL 2\n'
                    body += f'HM_NEW {key} {value}\nSTORE_LOCAL 0\nCALL churn\n'
                    body += 'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\n' + push_key + 'EQ\nASSERT\n'
                    body += 'LOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\n' + push_value + 'EQ\nASSERT\n'
                    helpers = ('.function keys 1 1 0 array 1\nLOAD_LOCAL 0\nHM_KEYS\nRET\n.end\n'
                               '.function values 1 1 0 array 1\nLOAD_LOCAL 0\nHM_VALUES\nRET\n.end\n')
                    self.paired(HEADER + '.function main 0 3 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n' + helpers + CHURN)

    def test_tagged_global_snapshots_keep_exact_pairs_through_calls(self):
        for key in (1, 5):
            for value in (1, 5):
                for through_call in (False, True):
                    with self.subTest(key=key, value=value, through_call=through_call):
                        push_key = 'PUSH_I64 -9223372036854775808\n' if key == 1 else 'PUSH_STR first\n'
                        push_value = 'PUSH_I64 -7\n' if value == 1 else 'PUSH_STR kept\n'
                        body = f'HM_NEW {key} {value}\nSTORE_GLOBAL 0\n'
                        for operation in ('keys', 'values'):
                            snapshot = 'CALL '+operation if through_call else 'HM_'+operation.upper()
                            body += 'LOAD_GLOBAL 0\n'+snapshot+'\nARR_LEN\nPUSH_I64 0\nEQ\nASSERT\n'
                        body += 'LOAD_GLOBAL 0\n'+push_key+push_value+'HM_SET\nPOP\n'
                        for slot, operation in enumerate(('keys', 'values')):
                            snapshot = 'CALL '+operation if through_call else 'HM_'+operation.upper()
                            body += 'LOAD_GLOBAL 0\n'+snapshot+f'\nSTORE_LOCAL {slot}\n'
                        body += f'HM_NEW {key} {value}\nSTORE_GLOBAL 0\nCALL churn\n'
                        for slot, tag, push in ((0, key, push_key), (1, value, push_value)):
                            body += f'LOAD_LOCAL {slot}\nDUP\nARR_LEN\nPUSH_I64 1\nEQ\nASSERT\n'
                            body += 'PUSH_I64 0\nARR_GET\nDUP\n'+f'TYPE_CHECK {tag}\nASSERT\n'+push+'EQ\nASSERT\n'
                        helpers = ''
                        if through_call:
                            helpers = ('.function keys 1 1 0 array 1\nLOAD_LOCAL 0\nHM_KEYS\nRET\n.end\n'
                                       '.function values 1 1 0 array 1\nLOAD_LOCAL 0\nHM_VALUES\nRET\n.end\n')
                        self.paired(HEADER+'.function main 0 2 0 int 1\n'+body+
                            'PUSH_I64 0\nRET\n.end\n'+helpers+CHURN)

    def test_tagged_global_map_return_retains_snapshot_pairs(self):
        for key in (1, 5):
            for value in (1, 5):
                with self.subTest(key=key, value=value):
                    push_key = 'PUSH_I64 -9223372036854775808\n' if key == 1 else 'PUSH_STR first\n'
                    push_value = 'PUSH_I64 -7\n' if value == 1 else 'PUSH_STR kept\n'
                    body = f'HM_NEW {key} {value}\nSTORE_GLOBAL 0\n'
                    body += 'LOAD_GLOBAL 0\n'+push_key+push_value+'HM_SET\nPOP\n'
                    body += 'CALL receiver\nHM_KEYS\nSTORE_LOCAL 0\n'
                    body += 'CALL receiver\nHM_VALUES\nSTORE_LOCAL 1\n'
                    body += f'HM_NEW {key} {value}\nSTORE_GLOBAL 0\nCALL churn\n'
                    for slot, tag, push in ((0, key, push_key), (1, value, push_value)):
                        body += f'LOAD_LOCAL {slot}\nDUP\nARR_LEN\nPUSH_I64 1\nEQ\nASSERT\n'
                        body += 'PUSH_I64 0\nARR_GET\nDUP\n'+f'TYPE_CHECK {tag}\nASSERT\n'+push+'EQ\nASSERT\n'
                    helper = '.function receiver 0 0 0 hashmap 1\nLOAD_GLOBAL 0\nRET\n.end\n'
                    self.paired(HEADER+'.function main 0 2 0 int 1\n'+body+
                                'PUSH_I64 0\nRET\n.end\n'+helper+CHURN)

    def test_conflicting_global_map_pairs_do_not_guess_snapshot_storage(self):
        text = (HEADER+'.function main 0 0 0 int 1\nHM_NEW 1 1\nSTORE_GLOBAL 0\n'
                'HM_NEW 5 5\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nHM_KEYS\nARR_LEN\nPOP\n'
                'PUSH_I64 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-global-map-refusal-') as folder:
            work = Path(folder)
            assembly, module, output = (work/name for name in ('input.nasm', 'input.nvm', 'prior.c'))
            assembly.write_text(text)
            self.checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
            self.checked([ROOT/'bin/nano_vm', '--verify-only', module])
            self.checked([ROOT/'bin/nano_vm', module])
            output.write_bytes(b'I preserve the prior native output.\n')
            result = subprocess.run([ROOT/'bin/nvm2c', module, '-o', output],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('snapshot element type', result.stderr)
            self.assertEqual(output.read_bytes(), b'I preserve the prior native output.\n')

    def test_snapshot_allocation_prefixes_and_successful_teardown(self):
        evidence = Path(os.environ.get('NANO_SCALAR_MAP_AOT_EVIDENCE', ROOT / '.test_output/scalar-map-aot-prefixes'))
        evidence.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='snapshot-prefixes-', dir=evidence))
        print('I retain generated snapshot allocation evidence at', work, flush=True)
        counts = {}

        def execute(name, command, *, failure=False):
            started = time.monotonic()
            result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                    capture_output=True, text=True, timeout=90)
            (work / (name + '.stdout')).write_text(result.stdout)
            (work / (name + '.stderr')).write_text(result.stderr)
            with (work / 'status.jsonl').open('a') as log:
                log.write(json.dumps({'name': name, 'argv': list(map(str, command)),
                                     'returncode': result.returncode, 'seconds': time.monotonic() - started}) + '\n')
            self.assertNotIn('AddressSanitizer', result.stderr)
            self.assertNotIn('UndefinedBehaviorSanitizer', result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            if failure:
                self.assertEqual(result.returncode, -signal.SIGABRT, result.stdout + result.stderr)
                self.assertRegex(result.stderr, r'^I stopped at a native invariant in [A-Za-z0-9_]+ at generated C line [0-9]+\.\n$')
                self.assertEqual(result.stdout, '')
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stderr, '')
            return result.stdout

        for key in (1, 5):
            for value in (1, 5):
                with self.subTest(key=key, value=value):
                    identity = f'pair-{key}-{value}'
                    assembly, module, c_source = (work / (identity + suffix) for suffix in ('.nasm', '.nvm', '.c'))
                    push_key = 'PUSH_I64 -9223372036854775808\n' if key == 1 else 'PUSH_STR first\n'
                    push_value = 'PUSH_I64 -7\n' if value == 1 else 'PUSH_STR kept\n'
                    body = f'HM_NEW {key} {value}\nSTORE_LOCAL 0\n'
                    for opcode, push in [('HM_KEYS', push_key), ('HM_VALUES', push_value)]:
                        body += 'LOAD_LOCAL 0\n' + opcode + '\n' + push + 'ARR_PUSH\nPOP\n'
                    body += 'LOAD_LOCAL 0\n' + push_key + push_value + 'HM_SET\nPOP\n'
                    body += 'LOAD_LOCAL 0\nHM_KEYS\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nHM_VALUES\nSTORE_LOCAL 2\n'
                    body += 'LOAD_LOCAL 0\n' + push_key + 'HM_DELETE\nPOP\n'
                    body += 'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\n' + push_key + 'EQ\nASSERT\n'
                    body += 'LOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\n' + push_value + 'EQ\nASSERT\n'
                    assembly.write_text(HEADER + '.function main 0 3 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n')
                    execute(identity + '-assemble', [ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
                    execute(identity + '-verify', [ROOT / 'bin/nano_vm', '--verify-only', module])
                    execute(identity + '-vm', [ROOT / 'bin/nano_vm', module])
                    execute(identity + '-translate', [ROOT / 'bin/nvm2c', module, '-o', c_source])
                    generated = c_source.read_text()
                    needle = 'if (debt >= nmap_collection_budget) nmap_collect();'
                    self.assertEqual(generated.count(needle), 1)
                    generated = generated.replace(needle, '(void)debt; nmap_collect();')
                    probe = (ROOT / 'tests/nanoisa/native_map_alloc_probe.h').read_text()
                    instrumented = (probe + '\n#define main fixture_main\n' + generated +
                        '\n#undef main\n#undef malloc\n#undef calloc\n#undef realloc\n#undef free\n'
                        'int main(int argc,char **argv){(void)ntest_realloc;'
                        'if(argc>1)ntest_fail=(size_t)strtoull(argv[1],NULL,10);'
                        'if(argc>2)ntest_persistent=1;int status=fixture_main();'
                        'if(ntest_live||ntest_bytes)abort();'
                        'printf("%zu %zu %zu\\n",ntest_calls,ntest_live,ntest_bytes);return status;}\n')
                    probe_source = work / (identity + '-probe.c')
                    probe_source.write_text(instrumented)
                    for optimization in ('0', '2'):
                        name = identity + '-O' + optimization
                        native = work / name
                        execute(name + '-compile', [*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                            '-std=c11', '-g', '-O' + optimization, '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all', probe_source, '-lm', '-o', native])
                        measured = execute(name + '-measure', [native])
                        self.assertRegex(measured, r'^[0-9]+ 0 0\n$')
                        count, live, size = map(int, measured.split())
                        self.assertGreater(count, 0)
                        self.assertEqual((live, size), (0, 0))
                        counts[name] = count
                        (work / 'allocation-counts.json').write_text(json.dumps(counts, indent=2) + '\n')
                        for persistent in (False, True):
                            for prefix in range(count):
                                execute(f'{name}-denied-{int(persistent)}-{prefix}',
                                    [native, str(prefix)] + (['persistent'] if persistent else []), failure=True)
                        self.assertEqual(execute(name + '-retry', [native]), measured)
        print('I measured every generated scalar snapshot allocation prefix:', counts, flush=True)

    def test_known_wrong_pairs_and_nonmap_snapshots_preserve_output(self):
        invalid = ['HM_NEW 4 1\nPOP\n', 'HM_NEW 1 3\nPOP\n',
                   'HM_NEW 5 1\nPUSH_I64 7\nHM_HAS\nPOP\n',
                   'HM_NEW 1 1\nPUSH_STR first\nHM_HAS\nPOP\n',
                   'HM_NEW 1 1\nPUSH_I64 7\nPUSH_STR kept\nHM_SET\nPOP\n',
                   'PUSH_I64 7\nHM_KEYS\nPOP\n', 'PUSH_STR first\nHM_VALUES\nPOP\n']
        with tempfile.TemporaryDirectory(prefix='nano-map-aot-refusals-') as tmp:
            source, module, output = (Path(tmp) / name for name in ('input.nasm', 'input.nvm', 'prior.c'))
            for body in invalid:
                with self.subTest(body=body):
                    source.write_text(HEADER + '.function main 0 0 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n')
                    self.checked([ROOT / 'bin/nanoisa', 'asm', source, '-o', module])
                    output.write_text('I retain prior output.\n')
                    result = subprocess.run([ROOT / 'bin/nvm2c', module, '-o', output],
                                            capture_output=True, text=True, timeout=60)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('I ', result.stderr)
                    self.assertEqual(output.read_text(), 'I retain prior output.\n')
