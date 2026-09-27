"""I compile and replay the unchanged mixed VM corpus as isolated real C."""
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from tests import test_file_cyclic
from tests import test_record_array_execution
from tests import test_record_array_vm

ROOT = Path(__file__).resolve().parents[1]


class RecordArrayGenerated(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-record-array-generated-'))
        print(f'I retain generated artifacts at {cls.artifacts}', flush=True)
        cls.cc = shlex.split(os.environ.get('RECORD_GENERATED_CC', 'cc'))
        cls.flags = ['-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1', '-Wall', '-Wextra',
                     '-Werror', *shlex.split(os.environ.get('RECORD_GENERATED_CFLAGS', '')),
                     '-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME',
                     '-DNANO_RECORD_ARRAY_GENERATED_PRIVATE', '-Isrc', '-Isrc/nanoisa',
                     '-Itests/nanoisa']
        cls.vm_objects = shlex.split(os.environ['RECORD_GENERATED_VM_OBJECTS'])
        cls.query_objects = shlex.split(os.environ['RECORD_GENERATED_QUERY_OBJECTS'])
        cls.ldflags = shlex.split(os.environ.get('RECORD_GENERATED_LDFLAGS', '-lm -lcrypto -lffi -pthread'))
        cls.native_ldflags = shlex.split(os.environ.get('RECORD_GENERATED_NATIVE_LDFLAGS', '-lm -pthread'))
        cls.nm = shlex.split(os.environ.get('RECORD_GENERATED_NM', 'nm'))
        os.environ['LSAN_OPTIONS'] = ''
        (cls.artifacts / 'selection.json').write_text(json.dumps({
            'python': sys.executable, 'cc': cls.cc, 'flags': cls.flags,
            'vm_common_inputs': cls.vm_objects, 'query_common_inputs': cls.query_objects,
            'capture_rebuilt': test_record_array_vm.PROVIDERS,
            'query_rebuilt': test_record_array_execution.PROVIDERS,
            'native_only': ['record_array_generated_private.c', 'generated program', 'replay driver'],
            'ldflags': cls.ldflags, 'native_ldflags': cls.native_ldflags,
            'nm': cls.nm, 'LSAN_OPTIONS': '',
            'instrumentation': 'Selected query/VM capture providers and generated/runtime TUs; common providers retain setup attribution.'}, indent=2) + '\n')

    def providers(self, prefix, sources, inputs, observed=False, omit=()):
        names = {Path(source).stem for source in sources}
        retained = [path for path in inputs if Path(path).stem not in names]
        outputs = []
        for source in sources:
            if source in omit:
                continue
            output = self.artifacts / f'{prefix}-{Path(source).stem}.o'
            hooks = ['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if observed else []
            self.command(f'{prefix}-{Path(source).stem}-build',
                         [*self.cc, *self.flags, *hooks, '-c', source, '-o', str(output)])
            outputs.append(str(output))
        return [*outputs, *retained]

    def test_emission_allocation_and_bounds(self):
        objects = self.providers('emission', test_record_array_execution.PROVIDERS,
                                 self.query_objects, observed=True)
        executable = self.artifacts / 'emission'
        self.command('emission-build', [*self.cc, *self.flags,
            'tests/nanoisa/test_record_array_generated_emission.c',
            'tests/nanoisa/record_array_alloc.c', *objects, *self.ldflags, '-o', str(executable)])
        output = self.command('emission-run', [str(executable)])
        self.assertIn(b'all93 emission recipes and all256 decisions', output)
        self.assertIn(b'actual emission allocation positions in both modes with independent recovery', output)
        print(output.decode(errors='replace').strip(), flush=True)

    def test_unchanged_corpus_generated_c(self):
        objects = self.providers('capture', test_record_array_vm.PROVIDERS,
                                 self.vm_objects, omit=('src/nanovm/vm.c',))
        executable = self.artifacts / 'capture'
        self.command('capture-build', [*self.cc, *self.flags,
            'tests/nanoisa/test_record_array_generated_capture.c',
            'src/nanoisa/nvm2c_record_array_private.c',
            'tests/nanoisa/record_array_alloc.c', *objects, *self.ldflags, '-o', str(executable)])
        corpus = self.artifacts / 'corpus'
        corpus.mkdir()
        output = self.command('capture-run', [str(executable), str(corpus)])
        self.assertIn(b'all93 actual retired operations and all256 decisions', output)
        summary = json.loads((corpus / 'corpus-counts.json').read_text())
        self.assertGreater(summary['products'], 0)
        self.assertGreater(summary['actions'], 0)
        sources = [corpus / f'product-{i:04d}.c' for i in range(summary['products'])]
        replays = [corpus / f'product-{i:04d}.replay.c' for i in range(summary['products'])]
        original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [*sources, *replays]}
        (self.artifacts / 'corpus-sha256.json').write_text(json.dumps(original, indent=2) + '\n')
        for opt in ('-O0', '-O2'):
            for observed in (False, True):
                mode = opt[1:] + ('-observed' if observed else '-linked')
                runtime = self.artifacts / f'{mode}-runtime.o'
                testing = ['-DNMS_TESTING', '-DNRG_OBSERVED'] if observed else []
                hooks = ['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if observed else []
                self.command(f'{mode}-runtime-build', [*self.cc, *self.flags, opt, *testing, *hooks,
                    '-c', 'src/nanoisa/record_array_generated_private.c', '-o', str(runtime)])
                lifecycle = self.artifacts / f'{mode}-lifecycle'
                if not observed:
                    self.command(f'{mode}-lifecycle-build', [*self.cc, *self.flags, opt,
                        'tests/nanoisa/test_record_array_generated_lifecycle.c', str(runtime),
                        *self.native_ldflags, '-o', str(lifecycle)])
                    control = self.command(f'{mode}-lifecycle-run', [str(lifecycle)])
                    self.assertIn(b'ABI, wrong thread, BUSY, acquired finish, retained result and recovery', control)
                support = []
                if observed:
                    allocator = self.artifacts / f'{mode}-allocator.o'
                    self.command(f'{mode}-allocator-build', [*self.cc, *self.flags, opt,
                        '-c', 'tests/nanoisa/record_array_alloc.c', '-o', str(allocator)])
                    support.append(str(allocator))
                for i, (source, replay) in enumerate(zip(sources, replays)):
                    name = f'{mode}-{i:04d}'
                    product = self.artifacts / name
                    self.command(name + '-build', [*self.cc, *self.flags, opt, *testing,
                        str(source), str(replay), str(runtime), *support,
                        *self.native_ldflags, '-o', str(product)])
                    unresolved = self.command(name + '-nm', [*self.nm, '-u', str(product)])
                    for forbidden in (b'vm_core_execute', b'vm_record_array', b'isa_decode', b'nvm_prepare'):
                        self.assertNotIn(forbidden, unresolved)
                    if not observed:
                        result = self.command(name + '-run', [str(product)])
                        self.assertIn(b'generated replay observations; status 0', result)
                        continue
                    result = self.command(name + '-baseline', [str(product), '--fault-baseline'])
                    self.assertIn(b'generated replay observations; status 0', result)
                    rows = re.findall(rb'^NRG_FAULT_BASELINE calls=([0-9]+) peak=([0-9]+)$', result, re.M)
                    self.assertEqual(len(rows), 1)
                    calls, peak = map(int, rows[0])
                    self.assertGreater(calls, 0)
                    ranges = [(start, min(start + 16, calls)) for start in range(0, calls, 16)]
                    plan = {'product': name, 'allocation_calls': calls, 'baseline_peak': peak,
                            'modes': ['one-shot', 'persistent'], 'positions': ranges,
                            'command_bound_seconds': 240, 'complete': False, 'workers': []}
                    (self.artifacts / (name + '-fault-plan.json')).write_text(json.dumps(plan, indent=2) + '\n')
                    cursor = recoveries = 0
                    for start, end in ranges:
                        self.assertEqual(start, cursor)
                        worker = f'{name}-fault-{start:06d}-{end:06d}'
                        result = self.command(worker, [str(product), '--fault-range', str(start), str(end), str(calls)])
                        self.assertIn(b'generated replay observations; status 0', result)
                        rows = re.findall(rb'^NRG_FAULT_RANGE begin=([0-9]+) end=([0-9]+) calls=([0-9]+) modes=([0-9]+) recoveries=([0-9]+) peak=([0-9]+)$', result, re.M)
                        self.assertEqual(len(rows), 1)
                        actual = tuple(map(int, rows[0]))
                        self.assertEqual(actual, (start, end, calls, 2, 2 * (end - start), peak))
                        plan['workers'].append({'name': worker, 'begin': start, 'end': end,
                                                'modes': 2, 'recoveries': actual[4]})
                        cursor = end
                        recoveries += actual[4]
                    self.assertEqual(cursor, calls)
                    self.assertEqual(recoveries, 2 * calls)
                    plan.update(complete=True, recoveries=recoveries)
                    (self.artifacts / (name + '-fault-coverage.json')).write_text(json.dumps(plan, indent=2) + '\n')
                print(f'I passed {mode}: {len(sources)} actual generated products.', flush=True)
        final = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [*sources, *replays]}
        self.assertEqual(original, final)
        (self.artifacts / 'corpus-after-sha256.json').write_text(json.dumps(final, indent=2) + '\n')
        print(output.decode(errors='replace').strip(), flush=True)


if __name__ == '__main__':
    unittest.main()
