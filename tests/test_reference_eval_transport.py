"""I test direct reference-evaluator transport with benign child programs."""
from pathlib import Path
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def native_compiler():
    selected = os.environ.get('NANO_NATIVE_TEST_CC')
    if selected:
        return shlex.split(selected)
    homebrew = Path('/opt/homebrew/opt/llvm/bin/clang')
    if platform.system() == 'Darwin' and homebrew.is_file():
        return [str(homebrew)]
    return shlex.split(os.environ.get('CC', 'cc'))


def native_sdk_flags():
    if platform.system() != 'Darwin':
        return []
    sdk = os.environ.get('SDKROOT', '').strip()
    if not sdk:
        result = subprocess.run(
            ['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-path'],
            capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        sdk = result.stdout.strip()
    if not Path(sdk).is_dir():
        raise RuntimeError(f'I cannot find the selected macOS SDK: {sdk}')
    return ['-isysroot', sdk]


def leak_detection_for(identity):
    if sys.platform == 'darwin' and 'Apple clang version' in identity:
        return '0'
    return '1'


HARNESS = r'''
#include "nanocore_export.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
void *fixture_realloc(void *pointer, size_t size) {
    return getenv("NANO_TEST_REF_REJECT_GROW") ? NULL : realloc(pointer, size);
}
static pid_t owned_children[2];
static int owned_count, tracking_error;
static void track_child(pid_t child) {
    if (child <= 0 || owned_count >= 2) tracking_error = 1;
    else owned_children[owned_count++] = child;
}
int fixture_spawnp(pid_t *child, const char *file,
                   const posix_spawn_file_actions_t *actions,
                   const posix_spawnattr_t *attributes,
                   char *const arguments[], char *const environment[]) {
    int result = posix_spawnp(child, file, actions, attributes, arguments, environment);
    if (!result) track_child(*child);
    return result;
}
pid_t fixture_fork(void) {
    pid_t child = fork();
    if (child > 0) track_child(child);
    return child;
}
static int descriptors(void) {
    int count = 0;
    for (int fd = 0; fd < 256; ++fd) if (fcntl(fd, F_GETFD) >= 0) count++;
    return count;
}
int main(int argc, char **argv) {
    if (argc != 4) return 10;
    int status = 0, saved_output = -1, ready[2] = {-1, -1};
    pid_t sentinel = -1;
    char *text = NULL, *result = NULL;
    FILE *input = fopen(argv[1], "rb");
    if (!input || fseek(input, 0, SEEK_END)) { status = 11; goto cleanup; }
    long size = ftell(input);
    if (size < 0 || fseek(input, 0, SEEK_SET)) { status = 12; goto cleanup; }
    text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, input) != (size_t)size) { status = 13; goto cleanup; }
    text[size] = 0;
    fclose(input); input = NULL;
    if (!strcmp(argv[3], "foreign-child")) {
        if (pipe(ready)) { status = 19; goto cleanup; }
        sentinel = fork();
        if (sentinel < 0) { status = 19; goto cleanup; }
        if (!sentinel) {
            close(ready[0]);
            char byte = 'R';
            if (write(ready[1], &byte, 1) != 1) _exit(19);
            close(ready[1]);
            for (;;) pause();
        }
        close(ready[1]); ready[1] = -1;
        char byte;
        ssize_t count;
        do { count = read(ready[0], &byte, 1); } while (count < 0 && errno == EINTR);
        close(ready[0]); ready[0] = -1;
        if (count != 1 || byte != 'R') { status = 19; goto cleanup; }
    }
    if (!strcmp(argv[3], "closed")) {
        saved_output = dup(STDOUT_FILENO);
        if (saved_output < 0 || fcntl(saved_output, F_SETFD, FD_CLOEXEC) < 0) { status = 14; goto cleanup; }
        close(STDIN_FILENO); close(STDOUT_FILENO); close(STDERR_FILENO);
    }
    if (!strcmp(argv[3], "ignore-pipe")) signal(SIGPIPE, SIG_IGN);
    int before = descriptors(), repeats = !strcmp(argv[3], "repeat") ? 30 : 1;
    for (int i = 0; i < repeats; ++i) {
        char *next = nanocore_reference_eval(text, strcmp(argv[2], "-") ? argv[2] : NULL);
        if (i && (!next || !result || strcmp(next, result))) { free(next); status = 15; goto cleanup; }
        free(result);
        result = next;
        if (descriptors() != before) { status = 16; goto cleanup; }
        if (tracking_error || (next && owned_count != 2)) { status = 20; goto cleanup; }
        for (int child = 0; child < owned_count; ++child) {
            int child_status;
            pid_t reaped;
            do { reaped = waitpid(owned_children[child], &child_status, WNOHANG); } while (reaped < 0 && errno == EINTR);
            int wait_error = errno;
            if (reaped > 0 || (reaped < 0 && wait_error == ECHILD)) owned_children[child] = -1;
            if (reaped != -1 || wait_error != ECHILD) {
                status = 17; goto cleanup;
            }
        }
        owned_count = 0;
        if (sentinel > 0) {
            int child_status;
            pid_t reaped;
            do { reaped = waitpid(sentinel, &child_status, WNOHANG); } while (reaped < 0 && errno == EINTR);
            if (reaped != 0) {
                if (reaped > 0 || errno == ECHILD) sentinel = -1;
                status = 21; goto cleanup;
            }
        }
        if (!next) break;
    }
    if (!strcmp(argv[3], "ignore-pipe")) {
        struct sigaction current;
        if (sigaction(SIGPIPE, NULL, &current) || current.sa_handler != SIG_IGN) { status = 18; goto cleanup; }
    }
    status = result ? 0 : 3;
cleanup:
    if (input) fclose(input);
    for (int fd = 0; fd < 2; ++fd) if (ready[fd] >= 0) close(ready[fd]);
    for (int child = 0; child < owned_count; ++child) if (owned_children[child] > 0) {
        pid_t reaped;
        kill(owned_children[child], SIGKILL);
        do { reaped = waitpid(owned_children[child], NULL, 0); } while (reaped < 0 && errno == EINTR);
    }
    if (sentinel > 0) {
        int child_status;
        pid_t reaped;
        kill(sentinel, SIGTERM);
        do { reaped = waitpid(sentinel, &child_status, 0); } while (reaped < 0 && errno == EINTR);
        if ((reaped != sentinel || !WIFSIGNALED(child_status) || WTERMSIG(child_status) != SIGTERM) && (status == 0 || status == 3)) status = 22;
    }
    if (saved_output >= 0) { dup2(saved_output, STDOUT_FILENO); close(saved_output); }
    if (!status && result) fwrite(result, 1, strlen(result), stdout);
    free(result); free(text);
    return status;
}
'''

class ReferenceTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='nano-reference-transport-')
        cls.base = Path(cls.tmp.name)
        source = cls.base/'harness.c'; source.write_text(HARNESS)
        cls.binary = cls.base/'harness'
        linker = '-Wl,-dead_strip' if platform.system() == 'Darwin' else '-Wl,--gc-sections'
        cls.compiler = native_compiler()
        identity = subprocess.run(cls.compiler + ['--version'], capture_output=True,
                                  text=True, timeout=30)
        if identity.returncode:
            raise RuntimeError(identity.stdout + identity.stderr)
        cls.leak_detection = leak_detection_for(identity.stdout + identity.stderr)
        print(f'reference transport sanitizer compiler={" ".join(cls.compiler)} '
              f'detect_leaks={cls.leak_detection}')
        flags = cls.compiler + native_sdk_flags() + [
            '-std=c11','-D_POSIX_C_SOURCE=200809L','-O2','-Wall','-Wextra','-Werror',
            '-ffunction-sections','-fdata-sections','-fsanitize=address,undefined',
            '-fno-sanitize-recover=all','-I'+str(ROOT/'src')]
        # I replace allocation only in this test-compiled translation unit.
        object_file=cls.base/'exporter.o'
        for command in (flags+['-Drealloc=fixture_realloc',
                               '-Dposix_spawnp=fixture_spawnp','-Dfork=fixture_fork',
                               '-c',ROOT/'src/nanocore_export.c','-o',object_file],
                        flags+[source,object_file,linker,'-o',cls.binary]):
            result = subprocess.run([str(x) for x in command],capture_output=True,text=True,timeout=60)
            if result.returncode: raise RuntimeError(result.stdout+result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stub(self, directory, body):
        directory.mkdir(parents=True,exist_ok=True)
        stub = directory/'nanocore-ref'
        stub.write_text('#!'+sys.executable+'\nimport sys\n'+body)
        stub.chmod(0o755)
        return directory/'compiler'

    def run_eval(self, text, compiler, mode='once', path=None, success=True, fail_grow=False):
        source=self.base/'input.sexpr';source.write_text(text)
        env={**os.environ,'ASAN_OPTIONS':f'detect_leaks={self.leak_detection}:abort_on_error=1'}
        if path is not None:env['PATH']=str(path)
        if fail_grow:env['NANO_TEST_REF_REJECT_GROW']='1'
        result=subprocess.run([self.binary,source,str(compiler),mode],capture_output=True,
            text=True,env=env,timeout=30)
        self.assertEqual(result.returncode,0 if success else 3,result.stdout+result.stderr)
        self.assertNotIn('Sanitizer',result.stderr)
        self.assertNotIn('runtime error:',result.stderr)
        return result.stdout

    def test_quoted_expression_and_adjacent_spaced_path(self):
        compiler=self.stub(self.base/'adjacent directory','sys.stdout.write(sys.stdin.read())\n')
        expression='(EString "O\'Brien said \\"hello\\" beside a path")'
        self.assertEqual(self.run_eval(expression,compiler),expression)

    def test_path_lookup_and_trim(self):
        folder=self.base/'path directory'
        self.stub(folder,'sys.stdin.read()\nsys.stdout.write("answer\\r\\n\\n")\n')
        self.assertEqual(self.run_eval('(EInt 42)','-',path=folder),'answer')
        self.assertEqual(self.run_eval('(EInt 42)',self.base/'missing compiler',path=folder),'answer')

    def test_large_input_and_output_before_consumption(self):
        compiler=self.stub(self.base/'full duplex',
            'sys.stdout.write("p" * 262144)\nsys.stdout.flush()\nsys.stdout.write(sys.stdin.read())\n')
        expression='(EString "'+('ordinary quoted text ' * 30000)+'")'
        self.assertEqual(self.run_eval(expression,compiler),'p'*262144+expression)

    def test_empty_nonzero_and_missing_program(self):
        for name,body in [('empty','sys.stdin.read()\n'),
                          ('status','sys.stdin.read()\nsys.stdout.write("partial")\nsys.exit(7)\n')]:
            compiler=self.stub(self.base/name,body)
            self.run_eval('(EInt 42)',compiler,success=False)
        empty=self.base/'empty path';empty.mkdir()
        self.run_eval('(EInt 42)','-',path=empty,success=False)

    def test_early_exit_keeps_parent_signal_policy_and_cleans_children(self):
        compiler=self.stub(self.base/'early exit','sys.exit(0)\n')
        expression='(EString "'+('text ' * 100000)+'")'
        self.run_eval(expression,compiler,mode='ignore-pipe',success=False)
        self.run_eval(expression,compiler,success=False)

    def test_output_allocation_failure_reaps_both_owned_children(self):
        compiler=self.stub(self.base/'allocation cleanup',
            'sys.stdout.write("p" * 262144)\nsys.stdout.flush()\nsys.stdin.read()\n')
        expression='(EString "'+('ordinary input ' * 30000)+'")'
        self.run_eval(expression,compiler,success=False,fail_grow=True)

    def test_repeated_calls_and_closed_standard_descriptors(self):
        compiler=self.stub(self.base/'repeated calls','sys.stdin.read()\nsys.stdout.write("ok\\n")\n')
        self.assertEqual(self.run_eval('(EInt 42)',compiler,mode='repeat'),'ok')
        self.assertEqual(self.run_eval('(EInt 42)',compiler,mode='closed'),'ok')

    def test_caller_child_survives_success_and_missing_program(self):
        compiler=self.stub(self.base/'caller child','sys.stdin.read()\nsys.stdout.write("ok\\n")\n')
        self.assertEqual(self.run_eval('(EInt 42)',compiler,mode='foreign-child'),'ok')
        empty=self.base/'caller child empty path';empty.mkdir()
        self.run_eval('(EInt 42)','-',mode='foreign-child',path=empty,success=False)

if __name__ == '__main__':
    unittest.main()
