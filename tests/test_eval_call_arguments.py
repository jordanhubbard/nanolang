"""I retain valid large call arguments and check my actual carrier boundary."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def call_source(count, mode):
    parameters = ', '.join(f'arg_{i}: int' for i in range(count))
    types = ','.join(['int'] * count)
    signature = f'fn({types})->string'
    literals = ' '.join(str(i) for i in range(count))
    arguments = ' '.join(f'(tick {i})' for i in range(count))
    checks = '\n'.join(f'assert (== arg_{i} {i})' for i in range(count))
    callee = {'direct': 'take', 'lexical': 'selected', 'computed': '(choose)'}[mode]
    return f'''let mut calls: int = 0
let mut choices: int = 0
fn tick(value: int) -> int {{
 assert (== calls value)
 set calls (+ calls 1)
 return value
}}
shadow tick {{ set calls 0 assert (== (tick 0) 0) assert (== calls 1) }}
fn take({parameters}) -> string {{
 {checks}
 let retained: string = (str_concat "retained" "!")
 return retained
}}
shadow take {{ assert (str_equals (take {literals}) "retained!") }}
fn choose() -> {signature} {{ set choices (+ choices 1) return take }}
shadow choose {{ assert (str_equals ((choose) {literals}) "retained!") }}
fn main() -> int {{
 let selected: {signature} = take
 set calls 0
 set choices 0
 let answer: string = ({callee} {arguments})
 assert (== calls {count})
 assert (== choices {1 if mode == 'computed' else 0})
 assert (str_equals answer "retained!")
 assert (str_equals (take {literals}) "retained!")
 assert (str_equals answer "retained!")
 return 0
}}
shadow main {{ assert (== (main) 0) }}
'''


class EvalCallArguments(unittest.TestCase):
    def command(self, argv):
        result = subprocess.run(list(map(str, argv)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_valid_inline_boundary_and_large_direct_lexical_computed_calls(self):
        with tempfile.TemporaryDirectory(prefix='nano-call-arguments-') as tmp:
            directory = Path(tmp)
            for count in (16, 17, 64):
                for mode in ('direct', 'lexical', 'computed'):
                    with self.subTest(count=count, mode=mode):
                        path = directory / 'main.nano'
                        path.write_text(call_source(count, mode))
                        self.command([ROOT / 'bin/nano', path])
                        self.command([ROOT / 'bin/nanoc_c', path, '-o', directory / 'native'])
                        self.command([directory / 'native'])
                        self.command([ROOT / 'bin/nano_virt', path, '--run'])

    def test_large_call_propagates_handler_return_before_later_arguments(self):
        count = 18
        parameters = ', '.join(f'arg_{i}: int' for i in range(count))
        signature = 'fn(' + ','.join(['int'] * count) + ')->int'
        literals = ' '.join(['0'] * count)
        arguments = ' '.join(['0'] * 16 + ['(emit)', '(mark)'])
        for mode, callee in [('direct', 'take'), ('lexical', 'selected'), ('computed', '(choose)')]:
            source = f'''effect Stop {{ stop: int -> int }}
let mut calls: int = 0
let mut choices: int = 0
fn take({parameters}) -> int {{ set calls 100 return 42 }}
shadow take {{ assert (== (take {literals}) 42) }}
fn choose() -> {signature} {{ set choices (+ choices 1) return take }}
shadow choose {{ assert (== ((choose) {literals}) 42) }}
fn emit() -> int {{ return perform Stop.stop(7) }}
shadow emit {{ assert (== (exercise) 7) }}
fn mark() -> int {{ set calls (+ calls 1) return 0 }}
shadow mark {{ set calls 0 assert (== (mark) 0) assert (== calls 1) }}
fn send() -> int {{ let selected: {signature} = take return ({callee} {arguments}) }}
shadow send {{ assert (== (exercise) 7) }}
fn exercise() -> int {{
 set calls 0
 set choices 0
 let result = handle {{ (send) }} with {{ stop value -> {{ return value }} }}
 return 99
}}
shadow exercise {{ assert (== (exercise) 7) assert (== calls 0) }}
fn main() -> int {{
 assert (== (exercise) 7)
 assert (== calls 0)
 assert (== choices {1 if mode == 'computed' else 0})
 return 0
}}
shadow main {{ assert (== (main) 0) }}
'''
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='nano-call-return-') as tmp:
                path = Path(tmp) / 'main.nano'
                path.write_text(source)
                self.command([ROOT / 'bin/nano', path])
                self.command([ROOT / 'bin/nanoc_c', path, '-o', Path(tmp) / 'native'])
                self.command([Path(tmp) / 'native'])

    def test_failed_large_call_shadow_preserves_prior_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-call-refusal-') as tmp:
            directory = Path(tmp)
            path = directory / 'main.nano'
            source = call_source(17, 'direct')
            source = source.replace('shadow main { assert (== (main) 0) }',
                                    'shadow main { assert (== (main) 1) }')
            path.write_text(source)
            output = directory / 'prior'
            output.write_bytes(b'I retain prior output.\n')
            result = subprocess.run([ROOT / 'bin/nanoc_c', path, '-o', output],
                                    cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_bytes(), b'I retain prior output.\n')
            self.assertIn('shadow', result.stderr.lower())

    def test_actual_carrier_inline_heap_cleanup_and_denial(self):
        with tempfile.TemporaryDirectory(prefix='nano-call-carrier-') as tmp:
            directory = Path(tmp)
            source = directory / 'carrier.c'
            source.write_text(r'''#include "src/nanolang.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
static size_t live, allocations;
static int deny;
static void *allocate(size_t bytes) {
    allocations++;
    if (deny) return NULL;
    void *p = malloc(bytes);
    assert(p); live++; return p;
}
static void release(void *p) { assert(live); live--; free(p); }
#define EVAL_CALL_ARGUMENT_MALLOC allocate
#define EVAL_CALL_ARGUMENT_FREE release
#include "src/eval/call_arguments.h"
int main(int argc, char **argv) {
    Value small[EVAL_CALL_INLINE_ARGUMENTS];
    if (argc > 1) {
        if (strcmp(argv[1], "negative") == 0)
            (void)eval_call_arguments_acquire(small, -1);
        deny = 1;
        (void)eval_call_arguments_acquire(small, 17);
        return 99;
    }
    int counts[] = {0, 1, 16, 17, 64, 4096};
    for (size_t i = 0; i < sizeof counts / sizeof counts[0]; ++i) {
        int count = counts[i];
        size_t before = allocations;
        Value *args = eval_call_arguments_acquire(small, count);
        assert((args == small) == (count <= 16));
        assert(allocations == before + (count > 16));
        for (int j = 0; j < count; ++j) {
            memset(&args[j], 0, sizeof args[j]);
            args[j].type = VAL_INT; args[j].as.int_val = j;
        }
        for (int j = 0; j < count; ++j) assert(args[j].as.int_val == j);
        eval_call_arguments_release(args, small);
        assert(live == 0);
    }
    return 0;
}
''')
            binary = directory / 'carrier'
            self.command([*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                          '-std=c99', '-D_GNU_SOURCE', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                          '-I', ROOT, '-I', ROOT / 'src', source, '-o', binary])
            self.command([binary])
            for mode, diagnostic in [('negative', 'represent'), ('denial', 'allocate')]:
                result = subprocess.run([binary, mode], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(result.stderr, f"I cannot {diagnostic} this call's argument storage.\n")
            self.command([binary])
