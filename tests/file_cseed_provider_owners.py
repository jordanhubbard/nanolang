"""I inspect real provider ownership and run ordinary wrappers, never File services."""
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import sys

MODULES = {
    'compiler_support': ('modules/compiler_support/compiler_support.nano', 'Support',
                         'assert (== (Support.module_artifact "") "")'),
    'std_json': ('modules/std/json/json.nano', 'Json',
                 'let value: Json.Json = (Json.parse "{}")\n'
                 '    assert (Json.is_object value)\n    (Json.free value)'),
    'file_companion': ('modules/file_companion/file_companion.nano', 'Companion',
                       'let token: int = (Companion.open "0;")\n'
                       '    assert (> token 0)\n'
                       '    assert (== (Companion.number token (- 0 1) 0) 1)\n'
                       '    assert (Companion.destroy token)'),
    'file_source_catalog': ('modules/file_source_catalog/file_source_catalog.nano', 'Catalog',
                            'assert (== (Catalog.text 0 0 0 0) "nsi:nanolang/filesystem")\n'
                            '    assert (== (Catalog.number 2 0 0 0) 1)'),
    'nanoisa': ('modules/nanoisa/nanoisa.nano', 'Isa',
                'assert (== (Isa.load_print "") "")\n'
                '    assert (!= (Isa.last_error) "")'),
    'forth_see': ('modules/forth_see/forth_see.nano', 'Forth',
                  'unsafe {\n    let detail: string = (Forth.see "dup" "")\n'
                  '    assert (str_contains detail "SEE: cannot load")\n    }'),
    'cjson_provider': ('modules/cjson_provider/cjson_provider.nano', 'CJsonOwner', ''),
    'nsi_file_catalog_provider': ('modules/nsi_file_catalog_provider/nsi_file_catalog_provider.nano', 'PlanOwner', ''),
}
ANCHORS = {
    'cJSON_Parse': 'cjson_provider',
    'nl_file_catalog_interface': 'nsi_file_catalog_provider',
    'nl_file_source_catalog_string': 'file_source_catalog',
    'nl_utf8_validate': None,  # I require the final runtime, not any module aggregate.
    'nl_nanoisa_last_error': 'nanoisa',
    'nanoisa_load_file': 'nanoisa',
    'vm_decode_module': 'nanoisa',
}
WRAPPERS = dict(compiler_support='nlc_module_artifact', std_json='nl_json_parse',
                file_companion='nl_file_companion_open',
                file_source_catalog='nl_file_source_catalog_string',
                nanoisa='nl_nanoisa_last_error', forth_see='nl_forth_see',
                cjson_provider='cJSON_Parse',
                nsi_file_catalog_provider='nl_file_catalog_interface')


def source(order, forth_binary):
    imports = '\n'.join(f'module "{MODULES[name][0]}" as {MODULES[name][1]}' for name in order)
    checks = '\n    '.join(MODULES[name][2] for name in order if MODULES[name][2])
    if 'forth_see' in order:
        checks += ('\n    unsafe {\n    let actual: string = (Forth.see "dup" '
                   + json.dumps(str(forth_binary)) + ')\n'
                   '    assert (str_contains actual "ISA implementation of Forth word: dup")\n'
                   '    assert (str_contains actual "NanoISA block:")\n    }')
    return imports + '\nfn main() -> int {\n    ' + checks + '\n    return 0\n}\nshadow main { assert (== (main) 0) }\n'


def symbols(test, label, path, dynamic=False):
    darwin = sys.platform == 'darwin'
    flags = ['-gU'] if darwin else (['-D', '-g', '--defined-only'] if dynamic else ['-g', '--defined-only'])
    out, _ = test.command(label, ['nm', *flags, path])
    found = []
    for line in out.decode().splitlines():
        parts = line.split()
        if len(parts) < 3 or len(parts[-2]) != 1:
            continue
        name = parts[-1]
        if darwin and name.startswith('_'):
            name = name[1:]
        found.append(name)
    return Counter(found)


def run(test, root, selected):
    work = test.work / 'public-provider-owners'
    work.mkdir()
    # A real native host exports its canonical array/GC runtime. Python alone
    # does not. I provide only that host boundary, never another data module.
    host_source = work / 'runtime-host.c'
    host_source.write_text('''#include "runtime/dyn_array.h"
#include "runtime/gc.h"
#include <stdlib.h>
unsigned owner_host_array_abi(void) { return NANO_DYN_ARRAY_ABI_VERSION; }
static unsigned releases;
void owner_host_release_keys(DynArray *array) {
    for (int64_t i = 0; i < dyn_array_length(array); ++i)
        free(dyn_array_get_string(array, i));
    gc_release(array);
    ++releases;
}
unsigned owner_host_releases(void) { return releases; }
''')
    host = work / ('runtime-host.dylib' if sys.platform == 'darwin' else 'runtime-host.so')
    runtime_inputs = [root/'src/runtime'/name for name in ('dyn_array.c', 'gc.c', 'gc_struct.c')]
    test.command('owner-runtime-host-build', [*test.cc, *test.flags, '-fPIC',
                 '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                 host_source, *runtime_inputs, '-lm', '-o', host])
    host_symbols = symbols(test, 'owner-runtime-host-symbols', host, True)
    for anchor in ANCHORS:
        test.assertEqual(host_symbols[anchor], 0, anchor)
    for name in ('dyn_array_new', 'dyn_array_push_string_copy', 'gc_release', 'owner_host_array_abi'):
        test.assertEqual(host_symbols[name], 1, name)
    (work/'runtime-host-identity.json').write_text(json.dumps(dict(
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [host_source,*runtime_inputs]},
        image=str(host), sha256=hashlib.sha256(host.read_bytes()).hexdigest(),
        scope='canonical runtime host only; no moved provider anchors; ordinary owner gate'), indent=2)+'\n')
    manifest = root/'modules/forth_see/module.json'
    metadata = json.loads(manifest.read_text())
    expected_sources = [str((manifest.parent/item).resolve())
                        for item in metadata['c_sources']+metadata['shared_c_sources']]
    recipe = (root/'examples/Makefile').read_text()
    declaration = re.search(r'^FORTH_SEE_C_SOURCES\s*:?=\s*((?:[^\n]*\\\n)*[^\n]*)', recipe, re.M)
    test.assertIsNotNone(declaration)
    actual_sources = [str((root/'examples'/item).resolve())
                      for item in declaration.group(1).replace('\\\n', ' ').split()]
    test.assertEqual(actual_sources, expected_sources)
    test.assertEqual(len(actual_sources), len(set(actual_sources)))
    (work/'examples-provider-inputs.json').write_text(json.dumps(dict(
        sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in actual_sources},
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        recipe_sha256=hashlib.sha256((root/'examples/Makefile').read_bytes()).hexdigest()), indent=2)+'\n')
    import shlex
    test.command('owner-existing-forth-see', ['make', '-C', root/'examples', '-j2',
                 'CC='+shlex.join(test.cc), 'test-forth-see'], timeout=1800)
    forth_binary = root/'bin/nl_forth_interpreter_vm'
    example_products = [forth_binary, root/'build/test_forth_see', root/'modules/forth_see/.build/libforth_see.so']
    (work/'examples-products.json').write_text(json.dumps(
        {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in example_products}, indent=2)+'\n')
    for path in example_products:
        # I retain the independent recipe products inside the ordinary report tree too.
        import shutil
        shutil.copyfile(path, work/('examples-'+path.name))
    observer = work / 'compiler-observer.py'
    observer.write_text('#!' + sys.executable + '\nimport json,os,subprocess,sys\n'
        'fd=os.open(os.environ["OWNER_COMMAND_LOG"],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\n'
        'os.write(fd,(json.dumps(sys.argv[1:])+"\\n").encode());os.close(fd)\n'
        'sys.exit(subprocess.call(json.loads(os.environ["OWNER_REAL_CC"])+sys.argv[1:]))\n')
    observer.chmod(0o755)
    all_modules = list(MODULES)[:6]
    cases = [(name, [name]) for name in MODULES]
    cases += [('isa-forth', ['nanoisa', 'forth_see']),
              ('forth-isa', ['forth_see', 'nanoisa']),
              ('all', all_modules), ('all-reversed', list(reversed(all_modules)))]
    summary = []
    for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
        for name, order in cases:
            label = compiler + '-' + name
            directory = work / label
            directory.mkdir()
            path = directory / 'main.nano'
            path.write_text(source(order, forth_binary))
            exe = directory / 'program'
            cache = directory / 'cache'
            log = directory / 'commands.jsonl'
            env = dict(NANO_BUILD_CACHE=str(cache), CC=str(observer), NANO_CC=str(observer),
                       OWNER_COMMAND_LOG=str(log), OWNER_REAL_CC=json.dumps(test.cc),
                       NANO_SHADOW_TRACE='1')
            expected, inputs = selected(path)
            (directory / 'selected-inputs.json').write_text(json.dumps(dict(expected=expected, inputs=inputs), indent=2)+'\n')
            args = [root/'bin'/compiler, path, '-o', exe, '--keep-c']
            if compiler == 'nanoc_c':
                args += ['--verbose', '--llm-shadow-json', directory/'shadows.json']
            out, err = test.command(label+'-build', args, timeout=1800, extra=env)
            if compiler == 'nanoc_c':
                report = json.loads((directory/'shadows.json').read_text())
                test.assertTrue(report['completed'] and report['success'])
                test.assertEqual(report['failures'], [])
                test.assertEqual(report['test_count'], len(expected))
                raw = re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ', out+b'\n'+err, re.M)
            else:
                raw = re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$', err, re.M)
            normalized = [re.sub(r'^__nano_module_+[0-9]+_', '', item.decode()) for item in raw]
            test.assertCountEqual(normalized, expected)
            (directory/'selection.json').write_text(json.dumps(dict(raw=[x.decode() for x in raw], normalized=normalized, expected=expected), indent=2)+'\n')
            test.command(label+'-run', [exe])
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            final = [args for args in commands if '-o' in args and args[args.index('-o')+1] == str(exe)]
            test.assertEqual(len(final), 1)
            objects = [Path(arg) for arg in final[0] if arg.endswith('.o') and Path(arg).is_file()]
            test.assertEqual(len(objects), len(set(map(str, objects))))
            published = symbols(test, label+'-executable-symbols', exe)
            artifacts = []
            if compiler == 'nanoc_c':
                # I inspect actual final link inputs, not every historical cache generation.
                module_objects = {obj.stem: obj for obj in objects if obj.stem in MODULES}
                test.assertTrue(set(order) <= set(module_objects))
                tables = {}
                for module, obj in module_objects.items():
                    table = symbols(test, label+'-'+module+'-static-symbols', obj)
                    tables[module] = table
                    for anchor, owner in ANCHORS.items():
                        test.assertEqual(table[anchor], int(module == owner), (label, module, anchor))
                    test.assertEqual(table[WRAPPERS[module]], 1)
                    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
                    library = obj.parent / ('lib'+module+suffix)
                    test.assertTrue(library.is_file(), library)
                    exports = symbols(test, label+'-'+module+'-dynamic-symbols', library, True)
                    test.assertEqual(exports[WRAPPERS[module]], 1)
                    for anchor, owner in ANCHORS.items():
                        test.assertEqual(exports[anchor], int(module == owner), (label, module, anchor))
                    test.command(label+'-'+module+'-dependencies',
                                 ['otool', '-L', library] if sys.platform == 'darwin' else ['readelf', '-d', library])
                    # Each target loads in a fresh runtime-only host process.
                    test.command(label+'-'+module+'-dynamic-run',
                                 [sys.executable, '-m', 'tests.file_cseed_provider_owners', '--dynamic', module, library, forth_binary, host])
                    artifacts.extend([obj, library])
                if 'forth_see' in tables:
                    test.assertIn('nanoisa', tables)
                    test.assertFalse(set(tables['forth_see']) & set(tables['nanoisa']))
                for anchor, owner in ANCHORS.items():
                    if owner in module_objects:
                        test.assertEqual(published[anchor], 1, (label, anchor))
            else:
                # Native preparation may compile sources directly into the final command.
                # I retain that actual command and prove unique canonical source inputs.
                sources = [str(Path(arg).resolve()) for arg in final[0] if arg.endswith('.c')]
                test.assertEqual(len(sources), len(set(sources)))
            for anchor in ('nl_utf8_validate',):
                test.assertEqual(published[anchor], 1)
            artifacts += [exe]
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
            (directory/'actual-artifacts.json').write_text(json.dumps(dict(hashes=hashes, final_command=final[0]), indent=2)+'\n')
            summary.append(dict(compiler=compiler, case=name, imports=order, shadows=len(expected), artifacts=hashes))
    (work/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    test.assertEqual(len(summary), 36)


def dynamic(module, path, forth_binary, runtime_host):
    # I match ffi_loader's RTLD_LAZY rather than invent an eager runtime contract.
    host = ctypes.CDLL(str(runtime_host), mode=os.RTLD_GLOBAL | os.RTLD_LAZY)
    abi = host.owner_host_array_abi
    abi.restype = ctypes.c_uint
    abi.argtypes = []
    assert abi() == 2
    lib = ctypes.CDLL(str(path), mode=os.RTLD_LOCAL | os.RTLD_LAZY)
    def fn(name, result, args):
        call = getattr(lib, name); call.restype = result; call.argtypes = args; return call
    text, integer, pointer = ctypes.c_char_p, ctypes.c_int64, ctypes.c_void_p
    if module in ('cjson_provider', 'std_json'):
        direct = module == 'cjson_provider'
        value = fn('cJSON_Parse' if direct else 'nl_json_parse', pointer, [text])(
            b'{}' if direct else b'{"first":1,"second":2}')
        assert value
        assert fn('cJSON_IsObject' if direct else 'nl_json_is_object', ctypes.c_int if direct else integer, [pointer])(value) == 1
        keys = None if direct else fn('nl_json_object_keys', pointer, [pointer])(value)
        fn('cJSON_Delete' if direct else 'nl_json_free', None, [pointer])(value)
        if not direct:
            assert keys
            length = host.dyn_array_length; length.restype = integer; length.argtypes = [pointer]
            get = host.dyn_array_get_string; get.restype = text; get.argtypes = [pointer, integer]
            assert length(keys) == 2
            assert get(keys, 0) == b'first' and get(keys, 1) == b'second'
            release = host.owner_host_release_keys; release.restype = None; release.argtypes = [pointer]
            release(keys)
            count = host.owner_host_releases; count.restype = ctypes.c_uint; count.argtypes = []
            assert count() == 1
    elif module == 'nsi_file_catalog_provider':
        assert fn('nl_file_catalog_interface', text, [])() == b'nsi:nanolang/filesystem'
    elif module == 'file_source_catalog':
        assert fn('nl_file_source_catalog_string', text, [integer]*4)(0, 0, 0, 0) == b'nsi:nanolang/filesystem'
    elif module == 'compiler_support':
        assert fn('nlc_module_artifact', text, [text])(b'') == b''
    elif module == 'file_companion':
        token = fn('nl_file_companion_open', integer, [text, integer])(b'0;', 2)
        assert token > 0
        assert fn('nl_file_companion_number', integer, [integer]*3)(token, -1, 0) == 1
        assert fn('nl_file_companion_destroy', ctypes.c_bool, [integer])(token)
    elif module == 'nanoisa':
        assert fn('nl_nanoisa_load_print', text, [text])(b'') == b''
        assert fn('nl_nanoisa_last_error', text, [])()
        assert fn('nl_nanoisa_load_print', text, [text])(os.fsencode(forth_binary))
    elif module == 'forth_see':
        assert b'SEE: cannot load' in fn('nl_forth_see', text, [text, text])(b'dup', b'')
        actual = fn('nl_forth_see', text, [text, text])(b'dup', os.fsencode(forth_binary))
        assert b'ISA implementation of Forth word: dup' in actual
        assert b'NanoISA block:' in actual
    else:
        raise AssertionError(module)
    print('PASS isolated dynamic wrapper with canonical runtime host', module, flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 6 and sys.argv[1] == '--dynamic'
    dynamic(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
