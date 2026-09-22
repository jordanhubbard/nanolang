"""I qualify metadata ownership and actual cache fingerprints without providers."""
import copy
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from tests.test_file_cyclic import FileCyclic
ROOT = Path(__file__).resolve().parents[1]

class ModuleSdkAbi(unittest.TestCase):
    command = FileCyclic.command

    def setUp(self):
        self.artifacts = Path(tempfile.mkdtemp(prefix='nano-sdk-abi-metadata-'))
        self.serial = 0
        self.probe = Path(os.environ['SDK_ABI_PROBE'])
        self.module = self.artifacts / 'provider'
        self.module.mkdir()

    def run_checked(self, args):
        self.serial += 1
        return self.command(str(self.serial), list(map(str, args)))

    def schema(self):
        return {'version': 1, 'target': 'fixture-host',
                'types': [{'semantic': 'int', 'c_type': 'int64_t', 'kind': 'int'},
                          {'semantic': 'Packet', 'c_type': 'Packet', 'kind': 'record', 'fields': ['x', 'owner']},
                          {'semantic': 'array<Packet>', 'c_type': 'DynArray', 'kind': 'array', 'abi': 'dyn_array_v2', 'element': 1}],
                'functions': [{'name': 'read', 'symbol': 'provider_read', 'parameters': [2], 'result': 1}]}

    def probe_metadata(self, schema=None, raw=None, refused=False):
        metadata = {'name': 'fixture', 'version': '1', 'c_sources': []}
        if schema is not None:
            metadata['typed_abi'] = schema
        (self.module / 'module.json').write_text(raw if raw is not None else json.dumps(metadata))
        if refused:
            # The outer owning command supervises this child in the same group.
            # Sanitizer diagnostics cannot be mistaken for an expected refusal.
            script = ('import subprocess,sys; r=subprocess.run(sys.argv[1:],capture_output=True); '
                      'sys.stdout.buffer.write(r.stdout); sys.stderr.buffer.write(r.stderr); '
                      'assert r.returncode==1; '
                      'assert not any(x in r.stderr for x in (b"Sanitizer",b"runtime error:"))')
            self.run_checked([sys.executable, '-c', script, self.probe, 'typed-abi', self.module])
            return None
        output = self.run_checked([self.probe, 'typed-abi', self.module]).decode().splitlines()
        self.assertEqual(len(output), 2)
        exact, absent = map(int, output[1].split())
        self.assertGreater(exact, 0)
        self.assertGreater(absent, 0)
        return json.loads(output[0]), exact, absent

    def test_owned_parser_and_allocation_rollback(self):
        compiler = shlex.split(os.environ['SDK_ABI_CC'])
        flags = shlex.split(os.environ['SDK_ABI_CFLAGS'])
        flags = ['-std=c11' if flag.startswith('-std=') else flag for flag in flags]
        output = self.artifacts / 'owned'
        self.run_checked([*compiler, *flags, '-Isrc', 'tests/test_module_sdk_abi.c',
                          os.environ['SDK_ABI_CJSON'], *shlex.split(os.environ['SDK_ABI_LDFLAGS']), '-o', output])
        self.assertIn(b'owned typed ABI parsing, roundtrip and allocation rollback', self.run_checked([output]))

    def test_actual_fingerprint_and_refusals(self):
        absent, old, baseline = self.probe_metadata()
        self.assertIsNone(absent)
        self.assertEqual(old, baseline)
        schema = self.schema()
        canonical, first, old = self.probe_metadata(schema)
        self.assertEqual(canonical, schema)
        self.assertNotEqual(first, old)
        self.assertEqual(old, baseline)
        reordered = {key: schema[key] for key in reversed(schema)}
        reordered['types'] = [{key: row[key] for key in reversed(row)} for row in schema['types']]
        self.assertEqual(self.probe_metadata(reordered), (canonical, first, baseline))
        self.assertEqual(self.probe_metadata(canonical), (canonical, first, baseline))
        changes = [lambda s: s.update(target='another-host'),
                   lambda s: s['types'][1].update(c_type='OtherPacket'),
                   lambda s: s['types'][1]['fields'].reverse(),
                   lambda s: s['functions'][0].update(symbol='other_read'),
                   lambda s: s['functions'][0].update(result=0)]
        for change in changes:
            changed = copy.deepcopy(schema)
            change(changed)
            _, new, same_old = self.probe_metadata(changed)
            self.assertNotEqual(new, first)
            self.assertEqual(same_old, baseline)
        bad = [lambda s: s.update(version=2),
               lambda s: s.update(unknown=True),
               lambda s: s['types'][1].update(c_type='Packet *'),
               lambda s: s['types'][1].update(fields=['x', 'x']),
               lambda s: s['types'][1].update(fields=['x; injected']),
               lambda s: s['types'][2].update(element=3),
               lambda s: s['types'][2].update(element=1.5),
               lambda s: s['types'][2].update(abi='unknown_abi'),
               lambda s: s['functions'][0].update(result=-1),
               lambda s: s['functions'][0].update(parameters=[3]),
               lambda s: s['functions'].append(copy.deepcopy(s['functions'][0])),
               lambda s: s['types'].append(copy.deepcopy(s['types'][0])),
               lambda s: s.update(target='x' * 4097),
               lambda s: s.update(types=[{'semantic': str(i), 'c_type': 'T', 'kind': 'int'} for i in range(4097)])]
        for change in bad:
            changed = copy.deepcopy(schema)
            change(changed)
            self.probe_metadata(changed, refused=True)
        encoded = json.dumps(schema)
        self.probe_metadata(raw='{"typed_abi":' + encoded + ',"typed_abi":' + encoded + '}', refused=True)
        self.probe_metadata(raw='{"typed_abi":' + encoded.replace('"version": 1', '"version": 1,"version": 1') + '}', refused=True)
        self.probe_metadata(raw='{"typed_abi":' + encoded.replace('fixture-host', 'fixture\\u0000host') + '}', refused=True)

if __name__ == '__main__':
    unittest.main()
