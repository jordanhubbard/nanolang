"""I copy artifact strings before provider cleanup and preserve borrowed results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArtifactStringRelease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano-artifact-release-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def command(self, args, success=True, **kwargs):
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=90, **kwargs)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, str(args))
        return result

    def library(self, name, source, extra=()):
        c, library = self.work/(name+'.c'), self.work/(name+('.dylib' if sys.platform == 'darwin' else '.so'))
        c.write_text(source)
        self.command(['cc', '-std=c11', '-D_GNU_SOURCE', '-shared', '-fPIC', c, *extra, '-o', library])
        return library

    def provider(self, name):
        return self.library(name, r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
static int64_t allocations, releases, borrowed;
static const char *owned(void) {
    const char *value = LABEL;
    char *out = malloc(strlen(value) + 1);
    if (!out) return NULL;
    strcpy(out, value); allocations++; return out;
}
static void cleanup(const char *value) {
    releases++;
    free((void *)value);
    const char *path = getenv("NANO_ARTIFACT_RELEASE_MARKER");
    if (path) { FILE *f = fopen(path, "a"); if (f) { fprintf(f, "%lld %lld\n", (long long)allocations, (long long)releases); fclose(f); } }
}
const char *path_basename(const char *input) { (void)input; return owned(); }
void path_basename__nano_string_release_v1(const char *value) { cleanup(value); }
const char *path_join(const char *a, const char *b) { (void)a; (void)b; return owned(); }
void path_join__nano_string_release_v1(const char *value) { cleanup(value); }
const char *nl_nanoisa_last_error(void) { return owned(); }
void nl_nanoisa_last_error__nano_string_release_v1(const char *value) { cleanup(value); }
const char *path_normalize(const char *input) { (void)input; borrowed++; return "borrowed-literal"; }
int64_t file_delete(const char *key) {
    if (!strcmp(key, "allocations")) return allocations;
    if (!strcmp(key, "releases")) return releases;
    if (!strcmp(key, "borrowed")) return borrowed;
    return allocations - releases;
}
'''.replace('LABEL', json.dumps(name)))

    def imports(self, entries):
        return ''.join(f'.import {json.dumps(str(library))} {json.dumps(symbol)} {signature}\n.import_kind {i} artifact\n'
                       for i, (library, symbol, signature) in enumerate(entries))

    def module(self, text):
        source, module = self.work/'input.nasm', self.work/'input.nvm'
        source.write_text(text)
        self.command([ROOT/'bin/nanoisa', 'asm', source, '-o', module])
        self.command([ROOT/'bin/nano_vm', '--verify-only', module])
        return module

    def native(self, module, transform=None):
        source, binary = self.work/'generated.c', self.work/'native'
        self.command([ROOT/'bin/nvm2c', module, '-o', source])
        if transform:
            source.write_text(transform(source.read_text()))
        self.command(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source,
                      '-o', binary, '-lm', *(['-ldl'] if sys.platform.startswith('linux') else [])])
        return binary

    def paired(self, text):
        module = self.module(text)
        vm = self.command([ROOT/'bin/nano_vm', module])
        native = self.command([self.native(module)], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})
        self.assertEqual(vm.stdout, native.stdout)
        return module

    def count(self, index, key, expected):
        return f'PUSH_STR {key}\nCALL_EXTERN {index}\nPUSH_I64 {expected}\nI64_EQ\nASSERT\n'

    def test_provider_cleanup_counts_zero_one_two_args_and_borrowed(self):
        lib = self.provider('owned')
        imports = self.imports([(lib,'path_basename','string string'), (lib,'path_join','string string string'),
                                (lib,'nl_nanoisa_last_error','string'), (lib,'path_normalize','string string'),
                                (lib,'file_delete','int string')])
        body = ('CALL_EXTERN 2\nSTORE_LOCAL 0\nPUSH_STR input\nCALL_EXTERN 0\nSTORE_LOCAL 1\n'
                'PUSH_STR input\nPUSH_STR input\nCALL_EXTERN 1\nPOP\n'
                'PUSH_STR input\nCALL_EXTERN 3\nPUSH_STR borrowed\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nPUSH_STR expected\nEQ\nASSERT\nLOAD_LOCAL 1\nPUSH_STR expected\nEQ\nASSERT\n')
        body += self.count(4,'allocations',3)+self.count(4,'releases',3)+self.count(4,'live',0)+self.count(4,'borrowed_key',1)
        strings = '.string input "input"\n.string expected "owned"\n.string borrowed "borrowed-literal"\n'
        strings += ''.join(f'.string {key} "{value}"\n' for key,value in [('allocations','allocations'),('releases','releases'),('live','live'),('borrowed_key','borrowed')])
        self.paired(imports+strings+'.entry main\n.function main 0 2 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')

    def test_same_symbol_distinct_libraries_keep_release_owner(self):
        left, right = self.provider('left'), self.provider('right')
        imports = self.imports([(left,'path_basename','string string'), (right,'path_basename','string string'),
                                (left,'file_delete','int string'), (right,'file_delete','int string')])
        body = ('PUSH_STR input\nCALL_EXTERN 0\nSTORE_LOCAL 0\n'
                'PUSH_STR input\nCALL_EXTERN 1\nPUSH_STR right\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nPUSH_STR left\nEQ\nASSERT\n')
        for index in (2,3):
            body += self.count(index,'allocations',1)+self.count(index,'releases',1)+self.count(index,'live',0)
        strings = '.string input "input"\n.string left "left"\n.string right "right"\n'
        strings += ''.join(f'.string {key} "{key}"\n' for key in ('allocations','releases','live'))
        self.paired(imports+strings+'.entry main\n.function main 0 1 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')

    def test_real_fs_text_empty_missing_and_embedded_nul(self):
        # I compile the actual filesystem implementation and its runtime closure.
        lib = self.library('real_fs', '#include "'+str(ROOT/'modules/std/fs.c')+'"\n',
                           [ROOT/'obj/runtime/gc.o', ROOT/'obj/runtime/dyn_array.o', ROOT/'obj/runtime/gc_struct.o'])
        (self.work/'text').write_text('retained text')
        (self.work/'empty').write_bytes(b'')
        (self.work/'nul').write_bytes(b'a\0b')
        imports = self.imports([(lib,'file_read','string string')])
        strings = '.string expected "retained text"\n'
        body = ''
        for name in ('text','empty','missing','nul'):
            strings += f'.string {name} {json.dumps(str(self.work/name))}\n'
            body += f'PUSH_STR {name}\nCALL_EXTERN 0\nDUP\nTYPE_CHECK 5\nASSERT\n'
            body += 'PUSH_STR expected\nEQ\nASSERT\n' if name == 'text' else 'STR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n'
        self.paired(imports+strings+'.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')

    def path_cases(self):
        existing = self.work/'existing'
        existing.write_text('retained')
        return [
            ('path_normalize', ['/alpha/../beta//'], '/beta', [''], '.'),
            ('path_canonical', [str(existing)], str(existing.resolve()), [''], ''),
            ('path_join', ['/alpha', 'beta'], '/alpha/beta', ['', ''], ''),
            ('path_basename', ['/alpha/beta'], 'beta', [''], '.'),
            ('path_dirname', ['/alpha/beta'], '/alpha', [''], '.'),
            ('path_relpath', ['/root/x', '/root/y'], '../x', ['', ''], '.'),
        ]

    def test_real_path_providers_repeated_results_preserve_aliases(self):
        lib = self.library('real_paths', '#include "'+str(ROOT/'modules/std/fs.c')+'"\n',
                           [ROOT/'obj/runtime/gc.o', ROOT/'obj/runtime/dyn_array.o', ROOT/'obj/runtime/gc_struct.o'])
        cases = self.path_cases()
        imports = self.imports([(lib, name, 'string' + ' string'*len(args))
                                for name, args, _, _, _ in cases])
        strings, normal, empty = '', [], []
        for i, (_, args, expected, blank, empty_expected) in enumerate(cases):
            calls = []
            for label, values in [('normal', args), ('empty', blank)]:
                call = ''
                for j, value in enumerate(values):
                    key = f'{label}_{i}_{j}'
                    strings += f'.string {key} {json.dumps(value)}\n'
                    call += f'PUSH_STR {key}\n'
                calls.append(call + f'CALL_EXTERN {i}\n')
            normal.append(calls[0])
            empty.append(calls[1])
            strings += f'.string expected_{i} {json.dumps(expected)}\n'
            strings += f'.string empty_expected_{i} {json.dumps(empty_expected)}\n'
        body = ''.join(call + f'STORE_LOCAL {i}\n' for i, call in enumerate(normal))
        body += 'PUSH_I64 0\nSTORE_LOCAL 6\nloop:\nLOAD_LOCAL 6\nPUSH_I64 2000\nI64_LT_S\nJMP_FALSE done\n'
        for i in range(len(cases)):
            body += normal[i] + f'PUSH_STR expected_{i}\nEQ\nASSERT\n'
            body += empty[i] + f'PUSH_STR empty_expected_{i}\nEQ\nASSERT\n'
        body += 'LOAD_LOCAL 6\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 6\nJMP loop\ndone:\n'
        for i in range(len(cases)):
            body += f'LOAD_LOCAL {i}\nPUSH_STR expected_{i}\nEQ\nASSERT\n'
        self.paired(imports+strings+'.entry main\n.function main 0 7 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')

    def test_real_path_provider_allocation_failure_retains_null_contract(self):
        # I replace only this provider translation unit's allocating operations.
        source = ('#include <stdlib.h>\n#include <string.h>\n'
                  'static void *fixture_malloc(size_t size) { (void)size; return NULL; }\n'
                  'static char *fixture_strdup(const char *text) { (void)text; return NULL; }\n'
                  'static char *fixture_realpath(const char *path, char *out) { (void)path; (void)out; return NULL; }\n'
                  '#define malloc fixture_malloc\n#define strdup fixture_strdup\n#define realpath fixture_realpath\n'
                  '#include "'+str(ROOT/'modules/std/fs.c')+'"\n'
                  '#undef malloc\n#undef strdup\n#undef realpath\n')
        lib = self.library('paths_oom', source,
                           [ROOT/'obj/runtime/gc.o', ROOT/'obj/runtime/dyn_array.o', ROOT/'obj/runtime/gc_struct.o'])
        for name, args, _, _, _ in self.path_cases():
            with self.subTest(provider=name):
                strings, call = '', ''
                for i, value in enumerate(args):
                    strings += f'.string arg_{i} {json.dumps(value)}\n'
                    call += f'PUSH_STR arg_{i}\n'
                text = self.imports([(lib, name, 'string'+' string'*len(args))])+strings
                text += '.entry main\n.function main 0 0 0 int 1\n'+call+'CALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n'
                module = self.module(text)
                result = self.command([ROOT/'bin/nano_vm',module], success=False)
                self.assertIn('could not retain the provider string result', result.stderr)
                self.command([self.native(module)], success=False,
                             env={**os.environ, 'ASAN_OPTIONS':'detect_leaks=1'})

    def test_null_provider_result_is_released_once_then_refused(self):
        lib = self.library('null_result', r'''
#include <stdio.h>
#include <stdlib.h>
const char *path_basename(const char *input) { (void)input; return NULL; }
void path_basename__nano_string_release_v1(const char *value) {
    if (value) abort();
    FILE *f = fopen(getenv("NANO_ARTIFACT_RELEASE_MARKER"), "a");
    if (!f) abort();
    fputs("released-null\n", f); fclose(f);
}
''')
        text = self.imports([(lib,'path_basename','string string')])+'.string input "input"\n.entry main\n.function main 0 0 0 int 1\nPUSH_STR input\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n'
        module = self.module(text)
        binary = self.native(module)
        for name, command in [('vm',[ROOT/'bin/nano_vm',module]), ('native',[binary])]:
            marker = self.work/(name+'-null')
            self.command(command, success=False, env={**os.environ, 'ASAN_OPTIONS':'detect_leaks=1',
                                                       'NANO_ARTIFACT_RELEASE_MARKER':str(marker)})
            self.assertEqual(marker.read_text(), 'released-null\n')

    def test_dependency_cleanup_symbol_is_not_accepted_as_owner(self):
        dependency = self.library('other_image', r'''
#include <stdlib.h>
int cleanup_anchor(void) { return 0; }
void path_basename__nano_string_release_v1(const char *value) { (void)value; abort(); }
''')
        lib = self.library('provider_image', r'''
#include <stdio.h>
#include <stdlib.h>
extern int cleanup_anchor(void);
const char *path_basename(const char *input) {
    (void)input; (void)cleanup_anchor();
    FILE *f = fopen(getenv("NANO_ARTIFACT_RELEASE_MARKER"), "w");
    if (f) { fputs("called", f); fclose(f); }
    return "borrowed";
}
''', [dependency])
        text = self.imports([(lib,'path_basename','string string')])+'.string input "input"\n.entry main\n.function main 0 0 0 int 1\nPUSH_STR input\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n'
        module = self.module(text)
        binary = self.native(module)
        marker = self.work/'should-not-call'
        env = {**os.environ, 'ASAN_OPTIONS':'detect_leaks=1', 'NANO_ARTIFACT_RELEASE_MARKER':str(marker)}
        result = self.command([ROOT/'bin/nano_vm',module], success=False, env=env)
        self.assertIn("own image", result.stderr)
        self.assertFalse(marker.exists())
        self.command([binary], success=False, env=env)
        self.assertFalse(marker.exists())

    def test_vm_copy_allocation_failure_still_releases_provider(self):
        lib = self.provider('vm-failure')
        text = self.imports([(lib,'path_basename','string string')])+'.string input "input"\n.entry main\n.function main 0 0 0 int 1\nPUSH_STR input\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n'
        module = self.module(text)
        shim = self.work/'vm_ffi_failure.c'
        shim.write_text('#define vm_string_new fixture_string_new\n#include "'+str(ROOT/'src/nanovm/vm_ffi.c')+'"\n#undef vm_string_new\n'
                        'VmString *fixture_string_new(VmHeap *heap, const char *data, uint32_t length) { (void)heap; (void)data; (void)length; return NULL; }\n')
        binary = self.work/'vm-copy-failure'
        makefile = self.work/'failure.mk'
        # I use the existing build closure without rebuilding or replacing shared tools.
        makefile.write_text('.PHONY: artifact-copy-failure\nartifact-copy-failure:\n'
            '\t$(CC) $(CFLAGS) -D_GNU_SOURCE '+str(shim)+' '
            '$(filter-out $(OBJ_DIR)/nanovm/vm_ffi.o,$(NANOVM_OBJECTS)) '
            '$(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) '
            '$(OBJ_DIR)/nanovm/vmd_protocol.o $(OBJ_DIR)/nanovm/vmd_client.o '
            '$(OBJ_DIR)/nanovm/main.o $(LDFLAGS) $(EXPORT_DYNAMIC_LDFLAGS) -o '+str(binary)+'\n')
        self.command(['make','-s','-f','Makefile.gnu','-f',makefile,'artifact-copy-failure'], cwd=ROOT)
        marker = self.work/'vm-copy-release'
        result = self.command([binary,module], success=False,
                              env={**os.environ, 'NANO_ARTIFACT_RELEASE_MARKER':str(marker)})
        self.assertIn('could not retain the provider string result', result.stderr)
        self.assertEqual(marker.read_text(), '1 1\n')

    def test_real_fs_allocation_failure_preserves_empty_sentinel(self):
        # Only the provider's malloc is replaced; runtime objects retain their allocator.
        source = ('#include <stdlib.h>\nstatic void *fixture_malloc(size_t size) { (void)size; return NULL; }\n'
                  '#define malloc fixture_malloc\n#include "'+str(ROOT/'modules/std/fs.c')+'"\n#undef malloc\n')
        lib = self.library('fs_oom', source,
                           [ROOT/'obj/runtime/gc.o', ROOT/'obj/runtime/dyn_array.o', ROOT/'obj/runtime/gc_struct.o'])
        path = self.work/'ordinary-text'
        path.write_text('ordinary text')
        text = self.imports([(lib,'file_read','string string')])+f'.string input {json.dumps(str(path))}\n'
        text += '.entry main\n.function main 0 0 0 int 1\nPUSH_STR input\nCALL_EXTERN 0\nDUP\nTYPE_CHECK 5\nASSERT\nSTR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
        self.paired(text)

    def test_native_copy_allocation_failure_still_releases_provider(self):
        lib = self.provider('failure')
        text = self.imports([(lib,'path_basename','string string')])+'.string input "input"\n.entry main\n.function main 0 0 0 int 1\nPUSH_STR input\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n'
        module = self.module(text)
        def fail_only_copy(source):
            needle = 'nstr_owned *owner = malloc(bytes);'
            self.assertEqual(source.count(needle), 1)
            # I isolate the managed copy allocation, after the provider returns.
            return source.replace(needle, 'nstr_owned *owner = NULL;')
        binary = self.native(module, fail_only_copy)
        marker = self.work/'released'
        self.command([binary], success=False, env={**os.environ, 'ASAN_OPTIONS':'detect_leaks=1',
                                                   'NANO_ARTIFACT_RELEASE_MARKER':str(marker)})
        self.assertEqual(marker.read_text(), '1 1\n')


if __name__ == '__main__':
    unittest.main()
