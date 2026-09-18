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
HARNESS = r'''
#include "nanocore_export.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
void *fixture_realloc(void *pointer, size_t size) {
    return getenv("NANO_TEST_REF_REJECT_GROW") ? NULL : realloc(pointer, size);
}
static int descriptors(void) {
    int count = 0;
    for (int fd = 0; fd < 256; ++fd) if (fcntl(fd, F_GETFD) >= 0) count++;
    return count;
}
int main(int argc, char **argv) {
    if (argc != 4) return 10;
    FILE *input = fopen(argv[1], "rb");
    if (!input || fseek(input, 0, SEEK_END)) return 11;
    long size = ftell(input);
    if (size < 0 || fseek(input, 0, SEEK_SET)) return 12;
    char *text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, input) != (size_t)size) return 13;
    text[size] = 0;
    fclose(input);
    int saved_output = -1;
    if (!strcmp(argv[3], "closed")) {
        saved_output = dup(STDOUT_FILENO);
        if (saved_output < 0 || fcntl(saved_output, F_SETFD, FD_CLOEXEC) < 0) return 14;
        close(STDIN_FILENO); close(STDOUT_FILENO); close(STDERR_FILENO);
    }
    if (!strcmp(argv[3], "ignore-pipe")) signal(SIGPIPE, SIG_IGN);
    int before = descriptors(), repeats = !strcmp(argv[3], "repeat") ? 30 : 1;
    char *result = NULL;
    for (int i = 0; i < repeats; ++i) {
        char *next = nanocore_reference_eval(text, strcmp(argv[2], "-") ? argv[2] : NULL);
        if (i && (!next || !result || strcmp(next, result))) return 15;
        free(result);
        result = next;
        if (descriptors() != before) return 16;
        int status;
        if (waitpid(-1, &status, WNOHANG) != -1 || errno != ECHILD) return 17;
        if (!next) break;
    }
    if (!strcmp(argv[3], "ignore-pipe")) {
        struct sigaction current;
        if (sigaction(SIGPIPE, NULL, &current) || current.sa_handler != SIG_IGN) return 18;
    }
    if (saved_output >= 0) { dup2(saved_output, STDOUT_FILENO); close(saved_output); }
    int status = result ? 0 : 3;
    if (result) fwrite(result, 1, strlen(result), stdout);
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
        flags = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC','cc')) + [
            '-std=c11','-D_POSIX_C_SOURCE=200809L','-O2','-Wall','-Wextra','-Werror',
            '-ffunction-sections','-fdata-sections','-fsanitize=address,undefined',
            '-fno-sanitize-recover=all','-I'+str(ROOT/'src')]
        # I replace allocation only in this test-compiled translation unit.
        object_file=cls.base/'exporter.o'
        for command in (flags+['-Drealloc=fixture_realloc','-c',ROOT/'src/nanocore_export.c','-o',object_file],
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
        env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:abort_on_error=1'}
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

if __name__ == '__main__':
    unittest.main()
