"""I retain exact nested-array record results across empty and nonempty returns."""
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import shlex
import sys
import tempfile
import time
import unittest

from tests.nanolang_translator_routes import ROOT, source_compiler_command
from tests.native_process import native_link_flags, run_captured


def source(nested, sliced, fallback):
    selected = '(at data.rows 0)' if nested else 'data.rows'
    literal = '[[7,8]]' if nested else '[7,8]'
    text = 'struct Data { rows: '+('array<array<int>>' if nested else 'array<int>')+' }\nstruct Reply { fields: array<int> }\n'
    if fallback:
        text += 'fn blank() -> Reply { return Reply { fields: [] } }\nshadow blank { assert (== (array_length (blank).fields) 0) }\n'
    text += 'fn select_rows(data: Data) -> Reply {\n'
    if fallback:
        text += ' if (== (array_length data.rows) 0) { return (blank) }\n'
    text += ' let values: array<int> = '+selected+'\n return Reply { fields: '+('(array_slice values 0 (array_length values))' if sliced else 'values')+' }\n}\n'
    text += 'shadow select_rows { assert (== (at (select_rows Data { rows: '+literal+' }).fields 0) 7)'
    if fallback:
        text += ' assert (== (array_length (select_rows Data { rows: [] }).fields) 0)'
    text += ' }\nfn main() -> int {\n let data: Data = Data { rows: '+literal+' }\n let answer: Reply = (select_rows data)\n assert (== (array_length answer.fields) 2)\n (array_set answer.fields 0 9)\n assert (== (at '+selected+' 0) '+('7' if sliced else '9')+')\n (println (at answer.fields 0))\n return 0\n}\nshadow main { assert (== (main) 0) }\n'
    if fallback:
        text = text.replace(' (println (at answer.fields 0))',
            ' assert (== (array_length (select_rows Data { rows: [] }).fields) 0)\n (println (at answer.fields 0))')
    return text


class NestedArrayResults(unittest.TestCase):
    def test_source_result_matrix_and_payload_refusals(self):
        work = Path(tempfile.mkdtemp(prefix='nano-nested-array-results-'))
        record = dict(scope='I require every valid translation and preserve prior output on failure. '
                      'I do not change compiler or runtime admission.', cases=[], commands=[])
        prior = b'I retain my prior output.\0\xff\n'

        def save():
            (work/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')

        def execute(command, label, *, build=False):
            index = len(record['commands']); started = time.monotonic()
            result = run_captured(command, timeout=600, build=build, cwd=ROOT)
            (work/f'{index:04d}.stdout').write_bytes(result.stdout)
            (work/f'{index:04d}.stderr').write_bytes(result.stderr)
            record['commands'].append(dict(label=label, command=list(map(str, command)), exit=result.returncode,
                seconds=time.monotonic()-started, stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
                stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
            save()
            return result

        producers = [('c-producer', [ROOT/'bin/nano_virt'])]
        native_source = source_compiler_command()
        if native_source:
            producers.append(('nanolang-producer', [*native_source, '--emit-nvm']))
        compiler = shlex.split(os.environ.get('CC', 'cc'))
        for nested, sliced, fallback in product((False, True), repeat=3):
            name = f'nested{int(nested)}-slice{int(sliced)}-fallback{int(fallback)}'
            path = work/(name+'.nano'); path.write_text(source(nested, sliced, fallback))
            case = dict(name=name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), producers=[])
            record['cases'].append(case); save()
            with self.subTest(case=name, route='c-seed-native'):
                legacy = work/(name+'-legacy')
                made = execute([ROOT/'bin/nanoc_c', path, '-o', legacy], name+'-legacy-build', build=True)
                self.assertEqual(made.returncode, 0, made.stderr.decode(errors='replace'))
                result = execute([legacy], name+'-legacy-run')
                self.assertEqual((result.returncode, result.stdout), (0, b'9\n'), result.stderr)
            for label, producer in producers:
                with self.subTest(case=name, route=label):
                    module = work/(name+'-'+label+'.nvm')
                    made = execute([*producer, path, '-o', module], name+'-'+label+'-build', build=True)
                    self.assertEqual(made.returncode, 0, made.stderr.decode(errors='replace'))
                    result = execute([ROOT/'bin/nano_vm', module, '--'], name+'-'+label+'-vm')
                    self.assertEqual((result.returncode, result.stdout), (0, b'9\n'), result.stderr)
                    output = module.with_suffix('.c'); output.write_bytes(prior)
                    translated = execute([ROOT/'bin/nvm2c', module, '-o', output], name+'-'+label+'-aot', build=True)
                    case['producers'].append(dict(label=label, module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                        aot_exit=translated.returncode, prior_preserved=output.read_bytes()==prior))
                    save()
                    if translated.returncode:
                        self.assertEqual(output.read_bytes(), prior)
                    self.assertEqual(translated.returncode, 0, translated.stderr.decode(errors='replace'))
                    executable = module.with_suffix('.native')
                    made = execute([*compiler, '-std=c11', '-O1', '-g', '-fsanitize=address,undefined',
                        '-fno-sanitize-recover=all', output, ROOT/'bin/nano_aot_runtime.o', '-lm',
                        *(['-ldl', '-Wl,--export-dynamic'] if sys.platform.startswith('linux') else []),
                        *native_link_flags(), '-o', executable], name+'-'+label+'-native-build', build=True)
                    self.assertEqual(made.returncode, 0, made.stderr.decode(errors='replace'))
                    result = execute([executable], name+'-'+label+'-native-run')
                    self.assertEqual((result.returncode, result.stdout), (0, b'9\n'), result.stderr)
        for nested in (False, True):
            name = f'wrong-element-kind-nested{int(nested)}'
            text = source(nested, True, True).replace('[[7,8]]', '[[true,false]]') if nested else source(False, True, True).replace('[7,8]', '[true,false]')
            path = work/(name+'.nano'); path.write_text(text)
            for label, producer in producers:
                with self.subTest(refused=name, route=label):
                    output = work/(name+'-'+label+'.nvm'); output.write_bytes(prior)
                    refused = execute([*producer, path, '-o', output], name+'-'+label, build=True)
                    self.assertNotEqual(refused.returncode, 0)
                    self.assertEqual(output.read_bytes(), prior)
