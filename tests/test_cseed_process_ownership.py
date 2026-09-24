"""I clean up raw C-seed maps without invalidating returned aliases."""
import os
import shlex
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.native_toolchain import native_cc

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ['-std=c99','-D_POSIX_C_SOURCE=200809L','-O1','-g','-Wall','-Wextra','-Werror',
         '-fsanitize=address,undefined','-fno-sanitize-recover=all','-pthread','-I'+str(ROOT/'src')]
ENV = {**os.environ,'ASAN_OPTIONS':'detect_leaks=1:detect_stack_use_after_return=1',
       'UBSAN_OPTIONS':'halt_on_error=1',
       'LSAN_OPTIONS':'use_globals=0:use_stacks=0:use_registers=0'}
HARNESS = r'''
#include "runtime/gc.h"
#include <assert.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
static _Atomic int finalized;
static int fail_allocate, fail_register;
void *probe_malloc(size_t size) {
    if (fail_allocate) { fail_allocate = 0; return NULL; }
    return malloc(size);
}
int probe_atexit(void (*fn)(void)) {
    if (fail_register) { fail_register = 0; return -1; }
    return atexit(fn);
}
static void dispose(void *ptr) {
    gc_process_forget(ptr);
    atomic_fetch_add(&finalized, 1);
    free(ptr);
}
static void checked_exit(void) {
    assert(gc_process_owned_count() == 0);
    assert(atomic_load(&finalized) == 4003);
}
static void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 1000; i++) {
        int *value = malloc(sizeof(*value)); assert(value); *value = i;
        assert(gc_process_own(value, dispose) == value);
        assert(gc_process_own(value, dispose) == value);
        assert(*value == i);
        if (i % 2) dispose(value);
    }
    return NULL;
}
int main(void) {
    assert(!atexit(checked_exit));
    fail_register = 1;
    assert(gc_process_own(malloc(1), dispose) == NULL);
    assert(gc_process_owned_count() == 0);
    fail_allocate = 1;
    assert(gc_process_own(malloc(1), dispose) == NULL);
    assert(gc_process_owned_count() == 0);
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) assert(!pthread_create(&threads[i], NULL, worker, NULL));
    for (int i = 0; i < 4; i++) assert(!pthread_join(threads[i], NULL));
    assert(gc_process_owned_count() == 2000);
    assert(atomic_load(&finalized) == 2002);
    gc_process_cleanup();
    assert(gc_process_owned_count() == 0);
    assert(atomic_load(&finalized) == 4002);
    assert(gc_process_own(malloc(1), dispose));
    return 0;
}
'''


class CSeedProcessOwnership(unittest.TestCase):
    def checked(self, command, **kwargs):
        result = subprocess.run(command,capture_output=True,text=True,timeout=120,**kwargs)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result

    def test_counted_cleanup_failure_concurrency_and_exit(self):
        with tempfile.TemporaryDirectory(prefix='nano-process-owner-') as tmp:
            work = Path(tmp)
            source, binary, obj = work/'probe.c', work/'probe', work/'gc.o'
            source.write_text(HARNESS)
            self.checked([*native_cc(),*FLAGS,'-Dmalloc=probe_malloc','-Datexit=probe_atexit',
                          '-c',ROOT/'src/runtime/gc.c','-o',obj])
            self.checked([*native_cc(),*FLAGS,source,obj,ROOT/'src/runtime/gc_struct.c','-o',binary])
            self.checked([binary],env=ENV)

    def test_generated_maps_preserve_aliases_and_explicit_free(self):
        with tempfile.TemporaryDirectory(prefix='nano-seed-map-owner-') as tmp:
            work = Path(tmp)
            source, binary = work/'main.nano', work/'program'
            source.write_text('fn make_values() -> HashMap<string, string> { '
                'let values: HashMap<string, string> = (map_new) '
                '(map_put values "key" "returned") return values }\n'
                'shadow make_values { let values: HashMap<string, string> = (make_values) '
                'assert (== (map_get values "key") "returned") (map_free values) }\n'
                'fn main() -> int { '
                'let first: HashMap<string, string> = (make_values) '
                'let alias: HashMap<string, string> = first '
                'let second: HashMap<string, string> = (make_values) '
                '(map_put alias "other" "retained") '
                'assert (== (map_get first "other") "retained") '
                '(map_free first) assert (== (map_get second "key") "returned") '
                'let ints: HashMap<string, int> = (map_new) (map_put ints "answer" 42) '
                'assert (== (map_get ints "answer") 42) return 0 }\n'
                'shadow main { assert true }\n')
            env = {**ENV,'NANO_CC':shlex.join(native_cc()),
                   'NANO_CFLAGS':'-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all',
                   'NANO_LDFLAGS':'-fsanitize=address,undefined'}
            self.checked([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,env=env)
            self.checked([binary],env=env)
            # I share the owner across separately compiled module code.
            module = work/'maps.nano'
            lines = source.read_text().splitlines()
            module.write_text('module Maps\n'+'pub '+lines[0]+'\n'+lines[1]+'\n')
            source.write_text('module "'+str(module)+'" as maps\n'+
                              '\n'.join(lines[2:]).replace('(make_values)', '(maps.make_values)')+'\n')
            self.checked([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,env=env)
            self.checked([binary],env=env)

    def test_schema_lists_preserve_nested_aliases_returns_and_explicit_free(self):
        with tempfile.TemporaryDirectory(prefix='nano-seed-list-owner-') as tmp:
            work = Path(tmp)
            source, binary = work/'main.nano', work/'program'
            schema = 'import "'+str(ROOT/'src_nano/generated/compiler_ast.nano')+'"\n'
            helper = ('fn make_block() -> ASTBlock { '
                'let items: List<ASTStmtRef> = (list_ASTStmtRef_with_capacity 1) '
                '(list_ASTStmtRef_push items ASTStmtRef { node_id: 7, node_type: 0 }) '
                '(list_ASTStmtRef_push items ASTStmtRef { node_id: 9, node_type: 0 }) '
                'return ASTBlock { node_type: 0, line: 1, column: 1, statements: items, '
                'is_par: false, is_flow: false } }\n'
                'shadow make_block { let block: ASTBlock = (make_block) '
                'assert (== (list_ASTStmtRef_length block.statements) 2) '
                'assert (not (list_ASTStmtRef_is_empty block.statements)) '
                '(list_ASTStmtRef_free block.statements) }\n')
            main = ('fn seed_list_ASTStmtRef_new() -> int { return 37 }\n'
                'shadow seed_list_ASTStmtRef_new { assert (== (seed_list_ASTStmtRef_new) 37) }\n'
                'fn main() -> int { assert (== (seed_list_ASTStmtRef_new) 37) '
                'let blocks: List<ASTBlock> = (list_ASTBlock_new) '
                'let block: ASTBlock = (make_block) (list_ASTBlock_push blocks block) '
                'let alias: List<ASTStmtRef> = block.statements '
                '(list_ASTStmtRef_push alias ASTStmtRef { node_id: 11, node_type: 0 }) '
                'let retained: ASTBlock = (list_ASTBlock_get blocks 0) '
                '(list_ASTBlock_free blocks) '
                'assert (== (list_ASTStmtRef_length retained.statements) 3) '
                'let item: ASTStmtRef = (list_ASTStmtRef_get alias 2) '
                'assert (== item.node_id 11) (list_ASTStmtRef_free alias) '
                'let final: ASTBlock = (make_block) '
                'assert (== (list_ASTStmtRef_length final.statements) 2) return 0 }\n'
                'shadow main { assert true }\n')
            env = {**ENV,'NANO_CC':shlex.join(native_cc()),
                   'NANO_CFLAGS':'-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all',
                   'NANO_LDFLAGS':'-fsanitize=address,undefined'}
            source.write_text(schema+helper+main)
            self.checked([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,env=env)
            self.checked([binary],env=env)
            module = work/'blocks.nano'
            module.write_text('module Blocks\n'+schema+'pub '+helper)
            source.write_text(schema+'module "'+str(module)+'" as blocks\n'+
                              main.replace('(make_block)', '(blocks.make_block)'))
            self.checked([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,env=env)
            self.checked([binary],env=env)

    def test_schema_list_capacity_argument_and_runtime_bounds(self):
        with tempfile.TemporaryDirectory(prefix='nano-list-capacity-') as tmp:
            work = Path(tmp)
            source, binary = work/'main.nano', work/'program'
            schema = 'import "'+str(ROOT/'src_nano/generated/compiler_ast.nano')+'"\n'
            for expression in ('(list_ASTStmtRef_new 1)',
                               '(list_ASTStmtRef_with_capacity)',
                               '(list_ASTStmtRef_with_capacity true)'):
                with self.subTest(expression=expression):
                    source.write_text(schema+'fn main() -> int { let xs: List<ASTStmtRef> = '+
                                      expression+' return 0 }\nshadow main { assert true }\n')
                    binary.write_text('prior output')
                    result = subprocess.run([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,
                                            capture_output=True,text=True,timeout=120)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('constructor arity',result.stdout+result.stderr)
                    self.assertEqual(binary.read_text(),'prior output')
            source.write_text('struct Point { value: int }\nfn main() -> int { '
                'let xs: List<Point> = (list_Point_with_capacity 1) return 0 }\n'
                'shadow main { assert true }\n')
            binary.write_text('prior output')
            result = subprocess.run([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,
                                    capture_output=True,text=True,timeout=120)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('schema-runtime list',result.stdout+result.stderr)
            self.assertEqual(binary.read_text(),'prior output')
            source.write_text(schema+'extern fn get_argv(index: int) -> string\nfn main() -> int { '
                'let mut mode: string = "" unsafe { set mode (get_argv 1) } '
                'let mut capacity: int = -1 '
                'if (== mode "zero") { set capacity 0 } '
                'if (== mode "large") { set capacity 2147483648 } '
                'let xs: List<ASTStmtRef> = (list_ASTStmtRef_with_capacity capacity) '
                'assert (list_ASTStmtRef_is_empty xs) '
                '(list_ASTStmtRef_push xs ASTStmtRef { node_id: 7, node_type: 0 }) '
                'assert (== (list_ASTStmtRef_length xs) 1) return 0 }\n'
                'shadow main { assert true }\n')
            env = {**ENV,'NANO_CC':shlex.join(native_cc()),
                   'NANO_CFLAGS':'-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all',
                   'NANO_LDFLAGS':'-fsanitize=address,undefined'}
            self.checked([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,env=env)
            self.checked([binary,'zero'],env=env)
            for argument in ('negative','large'):
                result = subprocess.run([binary,argument],env=env,capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('I cannot represent this list capacity.',result.stderr)
                self.assertNotIn('AddressSanitizer',result.stderr)
            source.write_text(schema+'fn main() -> int { return 0 }\n'
                'shadow main { let xs: List<ASTStmtRef> = (list_ASTStmtRef_with_capacity 2147483648) '
                'assert (== (list_ASTStmtRef_length xs) 0) }\n')
            binary.write_text('prior output')
            result = subprocess.run([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,
                                    capture_output=True,text=True,timeout=120)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(binary.read_text(),'prior output')
