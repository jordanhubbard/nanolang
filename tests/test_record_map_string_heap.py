"""Direct checked heap adapters; no selected VM execution."""
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_cyclic

class RecordMapStringHeap(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    def test_string_heap(self):
        self.artifacts = Path(tempfile.mkdtemp(prefix='nano-record-map-string-heap-'))
        print(f'I retain heap artifacts at {self.artifacts}', flush=True)
        cc = shlex.split(os.environ.get('RECORD_ARRAY_VM_CC', 'cc'))
        flags = [*shlex.split(os.environ.get('RECORD_ARRAY_VM_CFLAGS', '')),
                 '-std=c11', '-D_DEFAULT_SOURCE', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                 '-DNANO_RECORD_MAP_PRIVATE_RUNTIME', '-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME', '-DNANO_RECORD_MAP_PRIVATE_HEAP', '-Isrc']
        providers = ['src/nanovm/heap.c', 'src/nanovm/heap_cycles.c', 'src/nanovm/value.c',
                     'src/nanoisa/isa.c']
        (self.artifacts / 'selection.json').write_text(json.dumps({'cc': cc, 'flags': flags,
            'providers': providers}, indent=2) + '\n')
        products = []
        for source in providers:
            obj = self.artifacts / (Path(source).stem + '.o')
            compiled = source
            if source.endswith('/heap.c'):
                whitebox = self.artifacts / 'heap-whitebox.c'
                whitebox.write_text('#include ' + json.dumps(str(Path(source).resolve())) + '\n'
                    'bool test_shared_slot(const VmHashMap *m,NanoValue k,uint32_t *s,bool *f)'
                    '{return hm_find_slot(m,k,s,f);}\n')
                compiled = str(whitebox)
            self.command(Path(source).stem + '-build', [*cc, *flags, '-DRA_ALLOC_WRAP',
                '-include', 'tests/nanoisa/record_array_alloc.h', '-c', compiled, '-o', str(obj)])
            products.append(str(obj))
        exe = self.artifacts / 'heap'
        self.command('harness-build', [*cc, *flags, 'tests/nanovm/test_record_map_string_heap.c',
            'tests/nanoisa/record_array_alloc.c', *products, '-lm', '-o', str(exe)])
        output = self.command('heap-run', [str(exe)])
        self.assertIn(b'no VM dispatch or admission', output)
        print(output.decode(), flush=True)

if __name__ == '__main__':
    unittest.main()
