"""I retain capture aliases and release direct C-seed buffers at exit."""
import os
import shlex
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.native_toolchain import native_cc

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
static _Atomic int live;
static int fail_allocate, fail_register;
static void *allocate(size_t size) {
    if (fail_allocate) return NULL;
    void *p = malloc(size);
    if (p) atomic_fetch_add(&live, 1);
    return p;
}
static void release(void *p) {
    if (p) { assert(atomic_fetch_sub(&live, 1) > 0); free(p); }
}
static int register_exit(void (*callback)(void)) {
    return fail_register ? -1 : atexit(callback);
}
#define malloc allocate
#define free release
#define atexit register_exit
#include "runtime/cseed_capture.h"
#undef malloc
#undef free
#undef atexit
static void checked_exit(void) { assert(atomic_load(&live) == 0); }
static void *worker(void *unused) {
    (void)unused;
    const char *first = nl_exec_capture("printf first");
    const char *second = nl_exec_capture("printf second");
    assert(!strcmp(first, "first"));
    assert(!strcmp(second, "second"));
    return NULL;
}
int main(int argc, char **argv) {
    assert(atexit(checked_exit) == 0);
    if (argc > 1) {
        fail_allocate = !strcmp(argv[1], "allocation");
        fail_register = !strcmp(argv[1], "registration");
        assert(!strcmp(nl_exec_capture("printf refused"), ""));
        assert(atomic_load(&live) == 0);
        fail_allocate = fail_register = 0;
    }
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) assert(!pthread_create(&threads[i], NULL, worker, NULL));
    for (int i = 0; i < 4; i++) assert(!pthread_join(threads[i], NULL));
    assert(atomic_load(&live) == 8);
    nano_seed_capture_release_all();
    assert(atomic_load(&live) == 0);
    assert(!strcmp(nl_exec_capture("printf retained"), "retained"));
    assert(!strcmp(nl_exec_capture(":"), ""));
    assert(atomic_load(&live) == 2);
    return 0;
}
'''


class CSeedCapture(unittest.TestCase):
    def test_aliases_failure_cleanup_concurrency_and_exit(self):
        with tempfile.TemporaryDirectory(prefix='nano-cseed-capture-') as tmp:
            source, binary = Path(tmp)/'probe.c', Path(tmp)/'probe'
            source.write_text(HARNESS)
            result = subprocess.run([*native_cc(), '-std=c99', '-D_POSIX_C_SOURCE=200809L',
                '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all', '-pthread', '-I'+str(ROOT/'src'),
                source, '-o', binary],capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for arguments in ([], ['allocation'], ['registration']):
                with self.subTest(arguments=arguments):
                    result = subprocess.run([binary,*arguments],capture_output=True,text=True,timeout=30,
                        env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:detect_stack_use_after_return=1',
                             'LSAN_OPTIONS':'use_stacks=0:use_registers=0','UBSAN_OPTIONS':'halt_on_error=1'})
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_cseed_generated_capture_product(self):
        with tempfile.TemporaryDirectory(prefix='nano-cseed-capture-product-') as tmp:
            source, binary = Path(tmp)/'probe.nano', Path(tmp)/'probe'
            source.write_text('extern fn nl_exec_capture(cmd: string) -> string\n'
                'fn main() -> int { unsafe { '
                'let first: string = (nl_exec_capture "printf first") '
                'let second: string = (nl_exec_capture "printf second") '
                'assert (== first "first") assert (== second "second") } return 0 }\n'
                'shadow main { assert true }\n')
            env = {**os.environ, 'NANO_CC':shlex.join(native_cc()),
                   'NANO_CFLAGS':'-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all',
                   'NANO_LDFLAGS':'-fsanitize=address,undefined',
                   'ASAN_OPTIONS':'detect_leaks=1:detect_stack_use_after_return=1',
                   'LSAN_OPTIONS':'use_stacks=0:use_registers=0','UBSAN_OPTIONS':'halt_on_error=1'}
            result = subprocess.run([ROOT/'bin/nanoc_c',source,'-o',binary],cwd=ROOT,
                                    env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result = subprocess.run([binary],env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
