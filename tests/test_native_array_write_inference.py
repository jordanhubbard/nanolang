"""I defer array-valued writes until caller storage facts converge."""
from pathlib import Path
import os
import shlex
import sys
import tempfile
import unittest

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]
TRANSLATOR = Path(os.environ.get('NANOLANG_NVM2C', ROOT / 'bin/nvm2c'))


def program(tag, operation, reverse=False, receiver=7, via_global=False):
    """I vary table order independently of the runtime call order."""
    worker = ('.function write 1 1 0 array 1\n.parameters write array\n'
              'LOAD_LOCAL 0\n' + ('PUSH_I64 0\n' if operation == 'SET' else '')
              + f'ARR_NEW {tag}\nARR_{operation}\nRET\n.end\n')
    main = ('.function main 0 1 0 int 1\n'
            + (f'ARR_NEW {tag}\nARR_LITERAL {receiver} 1\n'
               if operation == 'SET' and receiver == 7 else f'ARR_NEW {receiver}\n')
            + ('AGG_PACK 0 0 0 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nAGG_GET 0\n'
               if via_global else '')
            + 'CALL write\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nARR_LEN\n'
            'PUSH_I64 1\nI64_EQ\nASSERT\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nARR_LEN\n'
            'PUSH_I64 0\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
    return '.entry main\n' + (main + worker if reverse else worker + main)


def numeric_program(floating, reverse, selector, wrong, multi=False):
    """I keep literal producers independent of a dynamically tagged argument."""
    constant = 'PUSH_F64 2.5' if floating else 'PUSH_I64 65535'
    second_constant = 'PUSH_F64 3.5' if floating else 'PUSH_I64 65534'
    initial = 'PUSH_F64 1.25' if floating else 'PUSH_I64 7'
    provided = 'LOAD_LOCAL 1'
    if multi:
        choice = ('LOAD_LOCAL 2\nPUSH_I64 0\nI64_EQ\nJMP_FALSE next\n'
                  + constant + '\nJMP joined\nnext:\n'
                  'LOAD_LOCAL 2\nPUSH_I64 1\nI64_EQ\nJMP_FALSE provided\n'
                  + second_constant + '\nJMP joined\nprovided:\n' + provided + '\n')
        expected = constant if selector == 0 else second_constant if selector == 1 else initial
        bad_path = wrong and selector == 2
        selection = f'PUSH_I64 {selector}'
    else:
        first, second = (provided, constant) if reverse else (constant, provided)
        choice = 'LOAD_LOCAL 2\nJMP_FALSE other\n' + first + '\nJMP joined\nother:\n' + second + '\n'
        expected = constant if bool(selector) != reverse else initial
        bad_path = wrong and bool(selector) == reverse
        selection = f'PUSH_BOOL {selector}'
    assembly = f'''.entry main
.function write 3 3 0 array 1
.parameters write array {'float' if floating else 'int'} {'int' if multi else 'bool'}
LOAD_LOCAL 0
PUSH_I64 0
{choice}joined:
ARR_SET
RET
.end
.function main 0 0 0 int 1
{'PUSH_BOOL 1' if wrong else initial}
STORE_GLOBAL 0
{initial}
ARR_LITERAL {3 if floating else 1} 1
LOAD_GLOBAL 0
{selection}
CALL write
PUSH_I64 0
ARR_GET
{expected}
{'F64_EQ' if floating else 'I64_EQ'}
ASSERT
PUSH_I64 0
RET
.end
'''
    return assembly, bad_path


def nonempty_program(tag, operation, reverse=False, via_global=False, mismatch=False):
    """I observe element tags and contents after writing a complete child array."""
    payload, check = {
        1: ('PUSH_I64 -7', 'PUSH_I64 -7\nI64_EQ\nASSERT'),
        2: ('PUSH_U8 255', 'PUSH_U8 255\nEQ\nASSERT'),
        3: ('PUSH_F64 -3.5', 'PUSH_F64 -3.5\nF64_EQ\nASSERT'),
        4: ('PUSH_BOOL 1', 'PUSH_BOOL 1\nEQ\nASSERT'),
        5: ('PUSH_STR marker', 'PUSH_STR marker\nSTR_EQ\nASSERT'),
        7: ('PUSH_I64 -7\nARR_LITERAL 1 1',
            'PUSH_I64 0\nARR_GET\nPUSH_I64 -7\nI64_EQ\nASSERT'),
    }[tag]
    initial_tag = (3 if tag == 1 else 1) if mismatch else tag
    initial_array = (('PUSH_F64 1.25\n' if initial_tag == 3 else 'PUSH_I64 19\n')
                     + f'ARR_LITERAL {initial_tag} 1\n') if mismatch else f'ARR_NEW {initial_tag}\n'
    initial = (initial_array + 'ARR_LITERAL 7 1\n'
               if operation == 'SET' or mismatch else 'ARR_NEW 7\n')
    count = 2 if mismatch and operation == 'PUSH' else 1
    worker = ('.function write 1 1 0 array 1\n.parameters write array\n'
              'LOAD_LOCAL 0\n' + ('PUSH_I64 0\n' if operation == 'SET' else '')
              + payload + f'\nARR_LITERAL {tag} 1\nARR_{operation}\nRET\n.end\n')
    retained = ''
    if mismatch and operation == 'PUSH':
        retained = ('LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\n'
                    + ('PUSH_F64 1.25\nF64_EQ\nASSERT\n' if initial_tag == 3
                       else 'PUSH_I64 19\nI64_EQ\nASSERT\n'))
    main = ('.function main 0 1 0 int 1\n' + initial
            + ('AGG_PACK 0 0 0 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nAGG_GET 0\n'
               if via_global else '')
            + f'CALL write\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nARR_LEN\nPUSH_I64 {count}\nI64_EQ\nASSERT\n'
            + f'LOAD_LOCAL 0\nPUSH_I64 {count - 1}\nARR_GET\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\n'
            + retained
            + f'LOAD_LOCAL 0\nPUSH_I64 {count - 1}\nARR_GET\nPUSH_I64 0\nARR_GET\n'
            + check + '\nPUSH_I64 0\nRET\n.end\n')
    return '.entry main\n.string marker "kept"\n' + (main + worker if reverse else worker + main)


class ArrayWriteInference(unittest.TestCase):
    def command(self, args, *, build=False):
        result = run_captured(args, cwd=ROOT, timeout=60, build=build)
        self.assertEqual(result.returncode, 0,
                         f'{args}\n{result.stdout.decode(errors="replace")}\n'
                         + result.stderr.decode(errors='replace'))
        return result

    def assemble(self, work, assembly):
        source, module = work / 'input.nasm', work / 'input.nvm'
        source.write_text(assembly)
        self.command([ROOT / 'bin/nanoisa', 'asm', source, '-o', module])
        return module

    def test_late_caller_nested_storage(self):
        with tempfile.TemporaryDirectory(prefix='nano-array-write-inference-') as directory:
            work = Path(directory)
            for via_global in (False, True):
                for tag in (1, 2, 3, 4, 5, 6, 7, 8, 11, 15):
                    for operation in ('SET', 'PUSH'):
                        for reverse in (False, True):
                            with self.subTest(tag=tag, operation=operation, reverse=reverse, via_global=via_global):
                                module = self.assemble(work, program(tag, operation, reverse, via_global=via_global))
                                self.command([ROOT / 'bin/nano_vm', module])
                                source, native = work / 'out.c', work / 'out'
                                self.command([TRANSLATOR, module, '-o', source], build=True)
                                self.command([*shlex.split(os.environ.get('CC', 'cc')),
                                              '-std=c11', '-O1', '-g', '-fsanitize=address,undefined',
                                              '-fno-sanitize-recover=all', source,
                                              ROOT / 'bin/nano_aot_runtime.o', '-lm',
                                              *(['-ldl', '-Wl,--export-dynamic']
                                                if sys.platform.startswith('linux') else []),
                                              *native_link_flags(), '-o', native], build=True)
                                self.command([native])

    def test_nonempty_nested_values_and_payload_refusals(self):
        with tempfile.TemporaryDirectory(prefix='nano-array-write-values-') as directory:
            for tag in (1, 2, 3, 4, 5, 7):
                for operation in ('SET', 'PUSH'):
                    for reverse in (False, True):
                        for via_global in (False, True):
                            for mismatch in (False, True):
                                with self.subTest(tag=tag, operation=operation, reverse=reverse,
                                                  via_global=via_global, mismatch=mismatch):
                                    work = Path(directory) / f'{tag}-{operation}-{int(reverse)}-{int(via_global)}-{int(mismatch)}'
                                    work.mkdir()
                                    module = self.assemble(work, nonempty_program(tag, operation, reverse, via_global, mismatch))
                                    self.command([ROOT / 'bin/nano_vm', module])
                                    source = work / 'out.c'
                                    prior = b'I preserve my prior output.\n'
                                    source.write_bytes(prior)
                                    translated = run_captured([TRANSLATOR, module, '-o', source],
                                                              cwd=ROOT, timeout=60, build=True)
                                    # I retain exact and mixed-read refusals; a checked
                                    # tagged SET can replace the only observable child type.
                                    if mismatch and (not via_global or operation == 'PUSH'):
                                        self.assertNotEqual(translated.returncode, 0, translated.stderr)
                                        self.assertEqual(source.read_bytes(), prior)
                                        continue
                                    self.assertEqual(translated.returncode, 0, translated.stderr)
                                    native = work / 'out'
                                    self.command([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O1', '-g',
                                                  '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source,
                                                  ROOT / 'bin/nano_aot_runtime.o', '-lm',
                                                  *(['-ldl'] if sys.platform.startswith('linux') else []),
                                                  *native_link_flags(), '-o', native], build=True)
                                    self.command([native])

    def test_numeric_join_checks_only_executed_tagged_values(self):
        cases = [(floating, reverse, selector, wrong, False)
                 for floating in (False, True) for reverse in (False, True)
                 for selector in (0, 1) for wrong in (False, True)]
        cases += [(floating, False, selector, wrong, True)
                  for floating in (False, True) for selector in (0, 1, 2) for wrong in (False, True)]
        self.assertEqual(len(cases), 28)
        with tempfile.TemporaryDirectory(prefix='nano-numeric-join-') as directory:
            work = Path(directory)
            for floating, reverse, selector, wrong, multi in cases:
                with self.subTest(floating=floating, reverse=reverse, selector=selector, wrong=wrong, multi=multi):
                    assembly, bad_path = numeric_program(floating, reverse, selector, wrong, multi)
                    module = self.assemble(work, assembly)
                    vm = run_captured([ROOT / 'bin/nano_vm', module], cwd=ROOT, timeout=60)
                    self.assertEqual(vm.returncode == 0, not bad_path, vm.stderr)
                    source, binary = work / 'out.c', work / 'out'
                    self.command([TRANSLATOR, module, '-o', source], build=True)
                    self.command([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O1', '-g',
                                  '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source,
                                  ROOT / 'bin/nano_aot_runtime.o', '-lm',
                                  *(['-ldl'] if sys.platform.startswith('linux') else []),
                                  *native_link_flags(), '-o', binary], build=True)
                    native = run_captured([binary], cwd=ROOT, timeout=60)
                    self.assertEqual(native.returncode == 0, not bad_path, native.stderr)
                    if bad_path:
                        self.assertIn(b'nvalue_require_float' if floating else b'nvalue_require_int', native.stderr)

    def test_incompatible_caller_preserves_prior_source(self):
        with tempfile.TemporaryDirectory(prefix='nano-array-write-refusal-') as directory:
            work = Path(directory)
            for receiver in (1, 4, 5, 8):
                for operation in ('SET', 'PUSH'):
                    for reverse in (False, True):
                        with self.subTest(receiver=receiver, operation=operation, reverse=reverse):
                            module = self.assemble(work, program(1, operation, reverse, receiver))
                            source = work / 'prior.c'
                            prior = b'I preserve my prior output.\n'
                            source.write_bytes(prior)
                            result = run_captured([TRANSLATOR, module, '-o', source],
                                                  cwd=ROOT, build=True, timeout=60)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertEqual(source.read_bytes(), prior)

    def test_checked_record_array_fields(self):
        assembly = '''.entry main
.function main 0 1 0 int 1
ARR_NEW 8
PUSH_VOID
AGG_PACK 0 0 0 1
ARR_PUSH
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_I64 0
PUSH_VOID
AGG_PACK 0 0 0 1
ARR_SET
POP
PUSH_I64 0
RET
.end
'''
        with tempfile.TemporaryDirectory(prefix='nano-record-array-fields-') as directory:
            work = Path(directory)
            module = self.assemble(work, assembly)
            generated = work / 'generated.c'
            self.command([TRANSLATOR, module, '-o', generated], build=True)
            probe = work / 'probe.c'
            probe.write_text('''#define main original_main
#include "generated.c"
#undef main
#include <assert.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
int main(int argc, char **argv) {
#ifdef __linux__
    /* I keep expected traps out of the host crash collector. */
    assert(prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) == 0);
#endif
    static const unsigned carriers[] = {3, 5, 6, 10, 12, 14, 15, 16, 17};
    void *pointer = malloc(1);
    assert(pointer);
    nrec_t raw = {0}, boxed = {0};
    raw.n = boxed.n = 1;
    if (argc == 1) {
        for (size_t i = 0; i < sizeof carriers / sizeof *carriers; ++i) {
            unsigned storage = carriers[i];
            raw.k[0] = storage;
            raw.a[0] = (narr_t)pointer;
            raw.sa[0] = (nsarr_t)pointer;
            raw.ra[0] = (nrarr_t)pointer;
            nmap_value value = nrec_array_value(&raw, 0, storage);
            assert(value.kind == 7 && value.integer == storage && value.text == pointer);
            boxed.k[0] = 8; boxed.vk[0] = 7;
            boxed.f[0] = storage; boxed.s[0] = pointer;
            value = nrec_array_value(&boxed, 0, storage);
            assert(value.kind == 7 && value.integer == storage && value.text == pointer);
            for (size_t j = 0; j < sizeof carriers / sizeof *carriers; ++j) {
                boxed.f[0] = carriers[j];
                assert(nrec_field_storage_matches(&raw, &boxed, 0) == (i == j));
                assert(nrec_field_storage_matches(&boxed, &raw, 0) == (i == j));
            }
            boxed.f[0] = storage;
            for (unsigned tag = 0; tag < 16; ++tag) {
                if (tag == 7) continue;
                boxed.vk[0] = tag;
                assert(!nrec_field_storage_matches(&raw, &boxed, 0));
                assert(!nrec_field_storage_matches(&boxed, &raw, 0));
            }
            boxed.vk[0] = 0; boxed.f[0] = 0; boxed.s[0] = NULL;
            value = nrec_array_value(&boxed, 0, storage);
            assert(value.kind == 0 && !value.text);
            boxed.vk[0] = 7; boxed.f[0] = storage;
            assert(!nrec_field_storage_matches(&raw, &boxed, 0));
            assert(!nrec_field_storage_matches(&boxed, &raw, 0));
        }
        free(pointer);
        return 0;
    }
    unsigned mode = (unsigned)atoi(argv[1]);
    unsigned storage = (unsigned)atoi(argv[2]);
    unsigned actual = (unsigned)atoi(argv[3]);
    boxed.k[0] = 8; boxed.vk[0] = 7;
    boxed.f[0] = actual; boxed.s[0] = pointer;
    if (mode == 1) boxed.vk[0] = actual;
    if (mode == 2) boxed.s[0] = NULL;
    if (mode == 3) boxed.k[0] = actual;
    if (mode == 4) boxed.n = 0;
    (void)nrec_array_value(mode == 5 ? NULL : &boxed, 0, storage);
    free(pointer);
    return 0;
}
''')
            binary = work / 'probe'
            self.command([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O1', '-g',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all', probe,
                          ROOT / 'bin/nano_aot_runtime.o', '-lm',
                          *(['-ldl'] if sys.platform.startswith('linux') else []), *native_link_flags(),
                          '-o', binary], build=True)
            self.command([binary])
            carriers = (3, 5, 6, 10, 12, 14, 15, 16, 17)
            refusals = [(0, expected, actual) for expected in carriers for actual in carriers
                        if expected != actual]
            refusals += [(1, expected, tag) for expected in carriers for tag in range(16)
                         if tag not in (0, 7)]
            refusals += [(mode, expected, expected) for expected in carriers for mode in (2, 4, 5)]
            refusals += [(3, 3, tag) for tag in range(18) if tag not in (3, 8)]
            refusals += [(0, tag, tag) for tag in range(18) if tag not in carriers]
            for mode, expected, actual in refusals:
                with self.subTest(mode=mode, expected=expected, actual=actual):
                    result = run_captured([binary, mode, expected, actual], cwd=ROOT, timeout=60)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'native invariant', result.stderr)


if __name__ == '__main__':
    unittest.main()
