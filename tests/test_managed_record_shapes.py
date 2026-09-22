"""I qualify ordinary record facts independently of executable admission."""
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest
from tests.managed_probe_flags import compiler_command, compile_flags, link_flags

ROOT = Path(__file__).resolve().parents[1]
NO = 0xffffffff
TAGS = ['void', 'int', 'u8', 'float', 'bool', 'string', 'function', 'array', 'struct', 'enum']


def function(name, body, local_tags=(), arity=0, result=1):
    return (name, body, tuple(local_tags), arity, result)


def program(body, layouts=None, local_tags=(), helpers=(), authority=True, flags=None):
    # Each layout is (kind, [(field tag, nested global layout)]).
    layouts = layouts if layouts is not None else [(0, [(1, NO)])]
    functions = [function('main', body + '\nPUSH_I64 0\nRET', local_tags), *helpers]
    encoded = struct.pack('<I', len(layouts))
    for kind, fields in layouts:
        encoded += struct.pack('<BBHI', kind, 0, len(fields), NO)
        for tag, nested in fields:
            encoded += struct.pack('<B3xII', tag, nested, NO)
    ownership = struct.pack('<II', 1, len(layouts))
    ownership += bytes(flags if flags is not None else [1 if k == 0 else 0 for k, _ in layouts])
    ownership += bytes((-len(ownership)) % 4)
    ownership += struct.pack('<I', len(functions))
    for name, code, locals_, arity, result in functions:
        ownership += struct.pack('<HH', len(locals_), arity)
        for tag in (result, *locals_):
            ownership += struct.pack('<BBHI', tag, 0, 0, NO)
    text = '.string text "leaf"\n.string empty ""\n.entry main\n'
    text += '.types ' + ' '.join(str(sum(k == kind for k, _ in layouts)) for kind in (0, 3, 2)) + '\n'
    text += ''.join(f'.layouts "{encoded[i:i+512].hex()}"\n' for i in range(0, len(encoded), 512))
    if authority:
        text += ''.join(f'.ownership "{ownership[i:i+512].hex()}"\n' for i in range(0, len(ownership), 512))
    for name, code, locals_, arity, result in functions:
        text += f'.function {name} {arity} {len(locals_)} 0 {TAGS[result]} {int(result != 0)}\n{code}\n.end\n'
        if arity:
            text += '.parameters ' + name + ' ' + ' '.join(TAGS[t] for t in locals_[:arity]) + '\n'
    return text


class RecordShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='nano-record-shapes-')
        cls.work = Path(cls.temp.name)
        cls.probes = []
        objects = shlex.split(os.environ['NMA_LINK_OBJECTS'])
        for name, compiler, flags in [('gcc', 'cc', []), ('clang', 'clang',
            shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS', '')) +
            ['-fsanitize=address,undefined', '-fno-sanitize-recover=all'])]:
            probe = cls.work / name
            subprocess.run([*compiler_command(compiler), *compile_flags(), *flags, '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                '-DNMA_TESTING', ROOT/'src/nanoisa/managed_array_shapes.c',
                ROOT/'tests/nanoisa/test_managed_record_shapes.c', *objects,
                '-lm', '-lcrypto', *link_flags(), '-o', probe], cwd=ROOT, check=True, capture_output=True, text=True)
            cls.probes.append(probe)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def command(self, args, success=True):
        run = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode == 0, success, run.stdout + run.stderr)
        return run

    def analyze(self, text, status=0, vm=False, budget=None, admit=False):
        source = self.work/'input.nasm'
        source.write_text(text)
        outputs = []
        for probe in self.probes:
            run = self.command([probe, source, *([] if budget is None else [budget])])
            lines = run.stdout.splitlines()
            header = list(map(int, lines[0].split()))
            self.assertEqual(header[0], status, run.stdout)
            outputs.append(lines)
        self.assertEqual(outputs[0], outputs[1])
        origins = [list(map(int, s.split()[1:])) for s in outputs[0][2:] if s.startswith('O ')]
        fields = [list(map(int, s.split()[1:])) for s in outputs[0][2:] if s.startswith('F ')]
        if vm or admit:
            module = self.work/'input.nvm'
            self.command([ROOT/'bin/nanoisa', 'asm', source, '-o', module])
            if vm:
                self.command([ROOT/'bin/nano_vm', module])
            if admit:
                for tool in ['nvm2llvm', 'nvm2wasm']:
                    output = self.work/('record.ll' if tool == 'nvm2llvm' else 'record.wasm')
                    self.command([ROOT/'bin'/tool, module, '-o', output])
                    if tool == 'nvm2wasm':
                        self.assertEqual(self.command(['wasmtime', 'run', '--invoke', 'nano_entry', output]).stdout, '0\n')
                    else:
                        native = self.work/'record-native'
                        self.command(['clang', *shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS', '')),
                                      output, '-x', 'none', '-o', native])
                        self.command([native])
        return header, origins, fields

    def test_exact_identity_counted_construction_and_checked_admission(self):
        for constructor in ['STRUCT_LITERAL 0 1', 'AGG_PACK 0 0 0 1']:
            h, origins, fields = self.analyze(program('PUSH_I64 42\n' + constructor +
                '\nDUP\nPUSH_I64 43\nSTRUCT_SET 0\nPOP\nAGG_GET 0\nPUSH_I64 43\nEQ\nASSERT'), vm=True, admit=True)
            self.assertEqual(h[1:4], [1, 1, 2])
            self.assertEqual(origins[0][3:7], [0, 0, 0, 1])
            self.assertEqual(fields, [[1 << 1, 0, 0]])
            self.assertEqual(h[6], 1)  # Old array selector still refuses nominal metadata.

    def test_scalar_tags_string_fields_and_exact_float_bits(self):
        layouts = [(0, [(3, NO), (4, NO), (5, NO), (2, NO)])]
        body = ('PUSH_I64 -9223372036854775808\nF64_FROM_BITS\nPUSH_BOOL 1\nPUSH_STR text\nPUSH_U8 255\n'
                'STRUCT_LITERAL 0 4\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nAGG_GET 0\nF64_TO_BITS\n'
                'PUSH_I64 -9223372036854775808\nEQ\nASSERT\nLOAD_LOCAL 0\nAGG_GET 1\nASSERT\n'
                'LOAD_LOCAL 0\nAGG_GET 2\nPUSH_STR text\nSTR_EQ\nASSERT\n'
                'LOAD_LOCAL 0\nAGG_GET 3\nPUSH_U8 255\nEQ\nASSERT')
        _, _, fields = self.analyze(program(body, layouts, [8]), vm=True, admit=True)
        self.assertEqual(fields, [[1 << t, 0, 0] for t in (3, 4, 5, 2)])

    def test_resource_authority_preserves_normal_verifier_refusal(self):
        source = self.work/'resource.nasm'
        source.write_text(program('PUSH_I64 1\nSTRUCT_LITERAL 0 1\nPOP', flags=[3]))
        module = self.work/'resource.nvm'
        module.write_bytes(b'prior resource artifact')
        result = self.command([ROOT/'bin/nanoisa', 'asm', source, '-o', module], success=False)
        self.assertIn('ownership', result.stdout + result.stderr)
        self.assertEqual(module.read_bytes(), b'prior resource artifact')

    def test_empty_struct_new_and_exact_count_boundary(self):
        self.analyze(program('STRUCT_NEW 0\nPOP', [(0, [])]), vm=True)
        self.analyze(program('STRUCT_NEW 0\nPOP'), status=1)
        self.analyze(program('STRUCT_LITERAL 0 0\nPOP'), status=1)
        self.analyze(program('PUSH_I64 1\nPUSH_I64 2\nSTRUCT_LITERAL 0 2\nPOP'), status=1)

    def test_forward_origins_preserve_exact_declared_indices(self):
        layouts = [(0, [(8, 2)]), (0, [(1, NO)]), (0, [(1, NO)])]
        body = 'PUSH_I64 17\nSTRUCT_LITERAL 2 1\nSTRUCT_LITERAL 0 1\nAGG_GET 0\nAGG_GET 0\nPUSH_I64 17\nEQ\nASSERT'
        header, origins, fields = self.analyze(program(body, layouts), vm=True)
        self.assertEqual(header[1:3], [2, 2])
        self.assertEqual([row[3:7] for row in origins], [[2, 2, 0, 1], [0, 0, 1, 1]])
        self.assertEqual(fields, [[1 << 1, 0, 0], [1 << 8, 0, 1]])
        self.analyze(program(body.replace('STRUCT_LITERAL 2 1', 'STRUCT_LITERAL 1 1'), layouts), status=1)

    def test_nested_reads_and_shared_field_replacement(self):
        layouts = [(0, [(1, NO)]), (0, [(8, 0)])]
        body = ('PUSH_I64 7\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nPUSH_I64 9\nAGG_PACK 0 0 0 1\nAGG_SET 0\nPOP\n'
                'LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nPUSH_I64 9\nEQ\nASSERT')
        h, origins, fields = self.analyze(program(body, layouts, [8]), vm=True)
        self.assertEqual(h[1], 3)
        self.assertEqual(fields[1], [1 << 8, 0, (1 << 0) | (1 << 2)])

    def test_same_shape_different_nominal_and_interleaved_layout(self):
        layouts = [(3, []), (0, [(1, NO)]), (0, [(1, NO)]), (0, [(8, 1)])]
        good = 'PUSH_I64 7\nSTRUCT_LITERAL 0 1\nSTRUCT_LITERAL 2 1\nAGG_GET 0\nAGG_GET 0\nPOP'
        _, origins, _ = self.analyze(program(good, layouts), vm=True)
        self.assertEqual([o[4] for o in origins], [1, 3])
        self.analyze(program(good.replace('STRUCT_LITERAL 0 1', 'STRUCT_LITERAL 1 1'), layouts), status=1)

    def test_alias_join_requires_every_receiver_field(self):
        layouts = [(0, [(1, NO)]), (0, [])]
        bad = 'PUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 7\nSTRUCT_LITERAL 0 1\nJMP join\nother:\nSTRUCT_NEW 1\njoin:\nAGG_GET 0\nPOP'
        self.analyze(program(bad, layouts), status=1)
        self.analyze(program(bad.replace('STRUCT_NEW 1', 'PUSH_I64 9\nSTRUCT_LITERAL 0 1'), layouts), vm=True)

    def test_call_return_reordered_arguments_and_bottom(self):
        helpers = [function('create', 'LOAD_LOCAL 0\nSTRUCT_LITERAL 0 1\nRET', [1], 1, 8),
                   function('write', 'LOAD_LOCAL 1\nLOAD_LOCAL 0\nAGG_SET 0\nRET', [1, 8], 2, 8)]
        body = 'PUSH_I64 7\nCALL create\nSTORE_LOCAL 0\nPUSH_I64 9\nLOAD_LOCAL 0\nCALL write\nPOP\nLOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 9\nEQ\nASSERT'
        _, origins, fields = self.analyze(program(body, local_tags=[8], helpers=helpers), vm=True)
        self.assertEqual(len(origins), 1)
        self.assertEqual(origins[0][1], 1)
        self.assertEqual(fields, [[1 << 1, 0, 0]])

    def test_repeated_site_retains_distinct_nested_origins(self):
        layouts = [(0, [(1, NO)]), (0, [(8, 0)])]
        helpers = [function('wrap', 'LOAD_LOCAL 0\nSTRUCT_LITERAL 1 1\nRET', [8], 1, 8)]
        body = 'PUSH_I64 1\nSTRUCT_LITERAL 0 1\nCALL wrap\nPOP\nPUSH_I64 2\nSTRUCT_LITERAL 0 1\nCALL wrap\nAGG_GET 0\nAGG_GET 0\nPUSH_I64 2\nEQ\nASSERT'
        _, origins, fields = self.analyze(program(body, layouts, helpers=helpers), vm=True)
        self.assertEqual(len(origins), 3)
        self.assertEqual(fields[2], [1 << 8, 0, 3])

    def test_recursive_summary_and_global_initialization(self):
        helpers = [function('__init__', 'PUSH_I64 7\nSTRUCT_LITERAL 0 1\nSTORE_GLOBAL 0\nRET', result=0),
            function('recurse', 'LOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_TRUE base\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nCALL recurse\nRET\nbase:\nLOAD_LOCAL 0\nPUSH_I64 9\nAGG_SET 0\nRET', [8, 1], 2, 8)]
        body = 'LOAD_GLOBAL 0\nPUSH_I64 3\nCALL recurse\nPOP\nLOAD_GLOBAL 0\nAGG_GET 0\nPUSH_I64 9\nEQ\nASSERT'
        h, _, _ = self.analyze(program(body, helpers=helpers), vm=True)
        self.assertGreater(h[5], 0)  # Initial VOID remains an explicit runtime tag obligation.

    def test_loop_site_and_reentry_global_writes(self):
        layouts = [(0, [(1, NO)]), (0, [(8, 0)])]
        body = 'PUSH_I64 1\nSTRUCT_LITERAL 0 1\nSTORE_LOCAL 0\nPUSH_I64 2\nSTORE_LOCAL 1\nloop:\nLOAD_LOCAL 0\nSTRUCT_LITERAL 1 1\nSTORE_GLOBAL 0\nPUSH_I64 2\nSTRUCT_LITERAL 0 1\nSTORE_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nDUP\nSTORE_LOCAL 1\nJMP_TRUE loop\nLOAD_GLOBAL 0\nAGG_GET 0\nAGG_GET 0\nPUSH_I64 2\nEQ\nASSERT'
        _, _, fields = self.analyze(program(body, layouts, [8, 1]), vm=True)
        self.assertEqual(fields[1], [1 << 8, 0, 5])
        # A later global store may affect an earlier read on a subsequent entry.
        bad = 'LOAD_GLOBAL 0\nAGG_GET 0\nPOP\nSTRUCT_NEW 1\nSTORE_GLOBAL 0'
        helpers = [function('__init__', 'PUSH_I64 7\nSTRUCT_LITERAL 0 1\nSTORE_GLOBAL 0\nRET', result=0)]
        self.analyze(program(bad, [(0, [(1, NO)]), (0, [])], helpers=helpers), status=1)

    def test_unknown_authority_parameters_and_deferred_operations(self):
        good = 'PUSH_I64 7\nSTRUCT_LITERAL 0 1\nPOP'
        self.analyze(program(good, authority=False), status=1)
        self.analyze(program(good, flags=[0]), status=1)
        unused = function('unused', 'LOAD_LOCAL 0\nAGG_GET 0\nRET', [8], 1, 1)
        self.analyze(program(good, helpers=[unused]), status=1)
        self.analyze(program('PUSH_I64 1\nI64_INVERT\nPOP'), status=1)
        self.analyze(program('PUSH_STR text\nSTRUCT_LITERAL 0 1\nPOP'), status=1)

    def test_disjoint_array_origins_and_mixed_storage_boundary(self):
        body = 'ARR_NEW 1\nPUSH_I64 2\nARR_PUSH\nPOP\nPUSH_I64 7\nSTRUCT_LITERAL 0 1\nPOP\nARR_NEW 5\nPUSH_STR text\nARR_PUSH\nPOP'
        _, origins, _ = self.analyze(program(body), vm=True)
        self.assertEqual([o[0] for o in origins], [0, 1, 0])
        self.analyze(program('ARR_NEW 0\nPUSH_I64 7\nSTRUCT_LITERAL 0 1\nARR_PUSH\nPOP'), status=1)

    def test_origin_limit_and_combined_storage_cap(self):
        body = 'STRUCT_NEW 0\nPOP\n' * 64
        self.analyze(program(body, [(0, [])]), vm=True)
        self.analyze(program(body + 'STRUCT_NEW 0\nPOP', [(0, [])]), status=3)
        # Main has 256 locals + a 200-deep literal stack: stride456.
        # Its 2299 instruction states fit CELLS, but 1000 field summaries do not.
        body = ('PUSH_I64 1\n' * 200 + 'STRUCT_LITERAL 0 200\nPOP\n') * 5
        count = len(body.splitlines()) + 2  # final push+RET
        body += 'NOP\n' * (2298 - count)
        self.analyze(program(body, [(0, [(1, NO)] * 200)], [1] * 256), status=3)
        self.analyze(program(body[:-12], [(0, [(1, NO)] * 200)], [1] * 256))

    def test_allocation_failure_output_atomicity_and_recovery(self):
        text = program('PUSH_I64 7\nSTRUCT_LITERAL 0 1\nAGG_GET 0\nPOP')
        statuses = []
        for budget in range(13):
            # The private analysis owns one state, six function buffers,
            # a field buffer and two published-report buffers.
            status = 4 if budget < 10 else 0
            self.analyze(text, status=status, budget=budget)
            statuses.append(status)
        self.assertIn(4, statuses)
        self.analyze(text, vm=True, admit=True)


if __name__ == '__main__':
    unittest.main()
