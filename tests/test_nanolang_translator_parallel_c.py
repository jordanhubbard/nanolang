"""I compare actual simultaneous C transfers with independent register snapshots."""
from pathlib import Path
import itertools
import os
import random
import shlex
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags
from tests.test_nanolang_translator_module_verifier import assembled
from tests.test_nanolang_translator_stack_heights import ins
from tests.test_nanolang_translator_scalar_c import MASK, signed
from tests.test_nanolang_translator_binary64_c import PATTERNS

ROOT = Path(__file__).resolve().parents[1]


def corpus():
    cases = [('empty', [], [], [])]
    for tag in range(1, 4):
        for width in range(1, 6):
            for rank, order in enumerate(itertools.permutations(range(width))):
                cases.append((f'perm-{tag}-{width}-{rank}', list(range(width)), list(order), [tag]*width))
        for left, destination in enumerate((0, 1, (1 << 32)-1)):
            for right, source in enumerate((0, 1, (1 << 32)-1)):
                cases.append((f'endpoint-{tag}-{left}-{right}', [destination], [source], [tag]))
        cases.extend(((f'fanout-{tag}', [0,1,2,3], [3,3,3,3], [tag]*4),
                      (f'chain-{tag}', [0,1,2,3], [1,2,3,4], [tag]*4),
                      (f'wide-{tag}', list(range(257)), [*range(1,257),0], [tag]*257)))
    cases.append(('mixed', [0,1,2,3,4,5], [1,0,3,2,5,4], [1,1,2,2,3,3]))
    cases.extend((('length-d', [0],[],[]), ('length-s',[],[0],[]), ('length-t',[],[],[1]),
                  ('length-short',[0,1],[1],[1,1]), ('length-tags',[0,1],[1,0],[1]),
                  ('duplicate',[0,0],[1,2],[1,1]), ('source-conflict',[1,2],[0,0],[1,3]),
                  ('source-destination-conflict',[1,0],[0,2],[1,3]),
                  ('destination-source-conflict',[1,2],[0,1],[1,3])))
    for value in (-(1<<63), -1, 1<<32, (1<<63)-1):
        cases.extend(((f'bad-d-{value}',[value],[0],[1]), (f'bad-s-{value}',[0],[value],[1])))
    cases.extend((f'bad-t-{value}',[0],[1],[value]) for value in (-(1<<63),-1,0,4,255,(1<<63)-1))
    cases.extend((('late-source',list(range(65)),[*range(1,65),1<<32],[1]*65),
                  ('late-tag',list(range(65)),list(range(1,66)),[1]*64+[4]),
                  ('late-duplicate',list(range(64))+[0],list(range(1,66)),[1]*65)))
    return cases


def register_types(destinations, sources, tags):
    if len(destinations) != len(sources) or len(destinations) != len(tags): return None
    if len(destinations) != len(set(destinations)): return None
    types = {}
    for destination, source, tag in zip(destinations, sources, tags):
        if tag not in (1,2,3): return None
        for register in (destination, source):
            if not 0 <= register < 1<<32: return None
            if register in types and types[register] != tag: return None
            types[register] = tag
    return types


def values_for(types, variant):
    rng = random.Random(1142 + variant)
    integers = (0, 1, MASK, 1<<63, (1<<63)-1, 0x0123456789abcdef, 0xfedcba9876543210)
    values = {}
    for index, (register, tag) in enumerate(sorted(types.items())):
        if tag == 2: value = (index + variant) % 2
        elif variant >= len(PATTERNS): value = rng.getrandbits(64)
        elif tag == 1: value = integers[(index + variant) % len(integers)]
        else: value = PATTERNS[(index + variant) % len(PATTERNS)]
        values[register] = value
    return values


def expected(destinations, sources, values, rounds):
    result = dict(values)
    for _ in range(rounds):
        snapshot = dict(result)
        result.update((destination, snapshot[source]) for destination, source in zip(destinations, sources))
    return result


def reports(report):
    assert report.startswith('PRELUDE-BEGIN\n')
    prelude, marker, rest = report[len('PRELUDE-BEGIN\n'):].partition('PRELUDE-END\n')
    assert marker == 'PRELUDE-END\n'
    rows = []
    while rest:
        header, newline, rest = rest.partition('\n')
        assert newline
        row = header.split('|')
        assert len(row) == 8 and row[0] == 'T'
        length = int(row[-1])
        text, rest = rest[:length], rest[length:]
        assert rest.startswith('END\n')
        rest = rest[len('END\n'):]
        rows.append((row, text))
    return prelude, rows


def execution_source(prelude, valid):
    lines = [prelude, '#include <stdio.h>', '#include <inttypes.h>', '#include <string.h>']
    inputs, wanted, widths = [], [], []
    for index, (_, destinations, sources, tags), text in valid:
        types = register_types(destinations, sources, tags)
        registers = sorted(types)
        widths.append((index, len(registers)))
        lines.extend((f'static void edge_case_{index}(const uint64_t *input, uint64_t *output, unsigned rounds) {{',
                      ' (void)input; (void)output; (void)rounds;'))
        for offset, register in enumerate(registers):
            c_type = 'double' if types[register] == 3 else 'int64_t'
            lines.append(f' {c_type} nano_aot_local_{register}; memcpy(&nano_aot_local_{register}, &input[{offset}], 8);')
        lines.extend((' if (rounds > 0) ' + text, ' if (rounds > 1) ' + text))
        for offset, register in enumerate(registers): lines.append(f' memcpy(&output[{offset}], &nano_aot_local_{register}, 8);')
        lines.append('}')
        for variant in range(32):
            values = values_for(types, variant)
            for rounds in (1, 2):
                inputs.append(f'{index} {rounds}' + ''.join(f' {values[r]:016x}' for r in registers) + '\n')
                result = expected(destinations, sources, values, rounds)
                wanted.extend(f'{result[r]:016x}\n' for r in registers)
    lines.extend(('int main(int argc, char **argv) {', ' if (argc != 2) return 2;',
                  ' FILE *file = fopen(argv[1], "r"); if (!file) return 2;',
                  ' unsigned index, rounds, width; uint64_t input[257], output[257]; int scanned;',
                  ' while ((scanned = fscanf(file, "%u %u", &index, &rounds)) == 2) {',
                  '  if (rounds < 1 || rounds > 2) { fclose(file); return 3; }', '  switch (index) {'))
    lines.extend(f'   case {index}: width = {width}; break;' for index, width in widths)
    lines.extend(('   default: fclose(file); return 3;', '  }',
                  '  for (unsigned i = 0; i < width; ++i) if (fscanf(file, "%" SCNx64, &input[i]) != 1) { fclose(file); return 4; }',
                  '  switch (index) {'))
    lines.extend(f'   case {index}: edge_case_{index}(input, output, rounds); break;' for index, _ in widths)
    lines.extend(('   default: fclose(file); return 3;', '  }',
                  '  for (unsigned i = 0; i < width; ++i) if (printf("%016" PRIx64 "\\n", output[i]) < 0) { fclose(file); return 5; }',
                  ' }', ' int failed = scanned != EOF || ferror(file);', ' if (fclose(file)) failed = 1;',
                  ' return failed ? 6 : 0;', '}'))
    return '\n'.join(lines)+'\n', ''.join(inputs), ''.join(wanted)


def reference_module(cases):
    bodies, local_counts, depths, wanted = [], [], [], []
    valid = [case for case in cases if register_types(*case[1:]) is not None]
    entry = b''.join(ins('CALL', index+1) + ins('POP') for index in range(len(valid))) + ins('PUSH_I64',0) + ins('RET')
    bodies.append(entry); local_counts.append(0); depths.append(1)
    for index, (_, destinations, sources, tags) in enumerate(valid):
        types = register_types(destinations, sources, tags)
        registers = sorted(types); dense = {r:i for i,r in enumerate(registers)}
        values = values_for(types, index % 32)
        code = bytearray()
        for register in registers:
            value, tag = values[register], types[register]
            if tag == 3: code.extend(bytes([2]) + struct.pack('<Q',value))
            else: code.extend(ins('PUSH_BOOL',value) if tag == 2 else ins('PUSH_I64',signed(value)))
            code.extend(ins('STORE_LOCAL',dense[register]))
        for _ in range(2):
            for source in sources: code.extend(ins('LOAD_LOCAL',dense[source]))
            for destination in reversed(destinations): code.extend(ins('STORE_LOCAL',dense[destination]))
        result = expected(destinations, sources, values, 2)
        for register in registers:
            code.extend(ins('LOAD_LOCAL',dense[register]))
            if types[register] == 3: code.extend(ins('F64_TO_BITS'))
            code.extend(ins('CAST_INT') + ins('PRINTLN'))
            wanted.append(str(signed(result[register]))+'\n')
        code.extend(ins('PUSH_I64',0) + ins('RET'))
        bodies.append(bytes(code));local_counts.append(len(registers));depths.append(max(1,len(destinations)))
    return assembled(bodies, returns=[[1]]*len(bodies), locals_=local_counts, declared=depths,
                     strings=(b'parallel_reference',)), ''.join(wanted)


class ParallelC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-parallel-c-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        cls.fixture = ROOT / 'tests/nvm2c_parallel_c_driver.nano'
        cls.routes = build_routes(cls.fixture, cls.directory)

    def test_actual_vm_and_reference_c(self):
        raw, wanted = reference_module(corpus())
        module, source, binary = (self.directory/n for n in ('parallel.nvm','parallel.c','parallel-reference'))
        module.write_bytes(raw)
        checked([ROOT/'bin/nano_vm','--verify-only',module])
        self.assertEqual(checked([ROOT/'bin/nano_vm',module,'--']),wanted)
        checked([ROOT/'bin/nvm2c',module,'-o',source],build=True)
        platform = ['-ldl','-Wl,--export-dynamic'] if os.uname().sysname == 'Linux' else []
        checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-O1','-g',
                 '-fsanitize=address,undefined','-fno-sanitize-recover=all',source,
                 ROOT/'bin/nano_aot_runtime.o','-lm',*platform,*native_link_flags(),'-o',binary],build=True)
        self.assertEqual(checked([binary]),wanted)

    def test_reports_and_actual_emitted_c(self):
        original = None
        cases = corpus()
        self.assertEqual(len(cases),523)
        self.assertEqual(sum(register_types(*c[1:]) is not None for c in cases),497)
        for route_name, route in self.routes:
            with self.subTest(route=route_name):
                report = checked(route)
                if original is None: original = report
                self.assertEqual(report,original)
                prelude, rows = reports(report)
                self.assertEqual(len(rows),len(cases))
                valid = []
                for index, ((header,text), case) in enumerate(zip(rows,cases)):
                    label,destinations,sources,tags = case
                    self.assertEqual(header[:5],['T',label,*[','.join(map(str,x)) for x in (destinations,sources,tags)]])
                    accepted = register_types(destinations,sources,tags) is not None
                    self.assertEqual(header[5],str(int(accepted)))
                    self.assertEqual(header[6],str(len(destinations) if accepted else 0))
                    if accepted: self.assertTrue(text.startswith('{\n') and text.endswith('}\n')); valid.append((index,case,text))
                    else: self.assertEqual(text,'')
                source, data, wanted = execution_source(prelude,valid)
                c_path, data_path = self.directory/(route_name+'.c'),self.directory/(route_name+'.txt')
                c_path.write_text(source);data_path.write_text(data)
                for mode,flags in [('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all']),('optimized',['-O3'])]:
                    binary = self.directory/(route_name+'-'+mode)
                    checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror',*flags,c_path,*native_link_flags(),'-o',binary],build=True)
                    self.assertEqual(checked([binary,data_path]),wanted)

    def test_imported_shadows(self):
        binary = self.directory/'imported-native'
        checked([ROOT/'bin/nanoc_c',self.fixture,'--test-imports','-o',binary],build=True)
        self.assertTrue(checked([binary]).startswith('PRELUDE-BEGIN\n'))
        producers = [('c',[ROOT/'bin/nano_virt'])]
        if source_compiler_command(): producers.append(('nanolang',source_compiler_command()))
        for name,command in producers:
            module = self.directory/('imported-'+name+'.nvm')
            checked([*command,self.fixture,'--test-imports','--emit-nvm','-o',module],build=True)
            checked([ROOT/'bin/nano_vm','--verify-only',module])
            self.assertTrue(checked([ROOT/'bin/nano_vm',module,'--']).startswith('PRELUDE-BEGIN\n'))
