"""I exercise real imported headers outside their declaring directories."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys


def run(test, root):
    work = test.work / 'dependency headers'
    work.mkdir()
    empty = work / 'unrelated cwd'
    empty.mkdir()
    # The outer runner retains durable output and process-group ownership. This
    # launcher changes only cwd before replacing itself with the actual driver.
    launch = work / 'launch.py'
    launch.write_text('import os,sys\nos.chdir(sys.argv[1])\nos.execv(sys.argv[2],sys.argv[2:])\n')
    wrappers = []
    directories = []
    for index, value in enumerate((7, 11)):
        leaf = work / f'leaf {index}'
        wrapper = work / f'wrapper {index}'
        include = work / f'include {index}'
        for directory in (leaf, wrapper, include):
            directory.mkdir()
        directories.extend((leaf.resolve(), include.resolve(), wrapper.resolve()))
        (include / f'value_{index}.h').write_text(
            f'#include <stdint.h>\nint64_t header_value_{index}(void);\n')
        (leaf / f'local_{index}.h').write_text(f'#define HEADER_VALUE_{index} {value}\n')
        (leaf / 'value.c').write_text(
            f'#include "value_{index}.h"\n#include "local_{index}.h"\n'
            f'int64_t header_value_{index}(void) {{ return HEADER_VALUE_{index}; }}\n')
        # A repeated canonical include must appear only once in each generated
        # module compile. I still retain its original metadata request order.
        (leaf / 'module.json').write_text(json.dumps(dict(
            name=f'header_leaf_{index}', headers=[f'value_{index}.h'],
            c_sources=['value.c'], include_dirs=[str(include), str(include / '.')]), indent=2)+'\n')
        source = leaf / 'leaf.nano'
        source.write_text(f'module header_leaf_{index}\npub extern fn header_value_{index}() -> int\n')
        selected = wrapper / 'wrapper.nano'
        selected.write_text(
            f'module header_wrapper_{index}\nmodule {json.dumps(str(source))} as Leaf\n'
            f'pub fn header_wrapper_{index}() -> int {{ unsafe {{ return (Leaf.header_value_{index}) }} }}\n'
            f'shadow header_wrapper_{index} {{ assert (== (header_wrapper_{index}) {value}) }}\n')
        wrappers.append(selected)
    observer = work / 'observer.py'
    observer.write_text('#!' + sys.executable + '\nimport json,os,subprocess,sys\n'
        'fd=os.open(os.environ["HEADER_COMMAND_LOG"],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\n'
        'os.write(fd,(json.dumps(sys.argv[1:])+"\\n").encode());os.close(fd)\n'
        'sys.exit(subprocess.call(json.loads(os.environ["HEADER_REAL_CC"])+sys.argv[1:]))\n')
    observer.chmod(0o755)
    for reverse in (False, True):
        order = (1, 0) if reverse else (0, 1)
        source = work / ('reverse.nano' if reverse else 'forward.nano')
        source.write_text('\n'.join(f'module {json.dumps(str(wrappers[i]))} as W{i}' for i in order)+
            '\nfn main() -> int { assert (== (+ (W0.header_wrapper_0) (W1.header_wrapper_1)) 18) return 0 }\n'
            'shadow main { assert (== (main) 0) }\n')
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            name = f'headers-{compiler}-{int(reverse)}'
            output = work / name
            commands = work / (name + '-cc.jsonl')
            shadows = work / (name + '-shadows.json')
            args = [sys.executable, launch, empty, root / 'bin' / compiler, source, '-o', output]
            if compiler == 'nanoc_c':
                args += ['--verbose', '--llm-shadow-json', shadows]
            out, err = test.command(name+'-build', args, timeout=900, extra=dict(
                CC=shlex.quote(str(observer)), NANO_CC=shlex.quote(str(observer)), HEADER_REAL_CC=json.dumps(test.cc),
                HEADER_COMMAND_LOG=str(commands), NANO_SHADOW_TRACE='1'))
            if compiler == 'nanoc_c':
                report = json.loads(shadows.read_text())
                test.assertTrue(report['completed']); test.assertTrue(report['success'])
                test.assertEqual(report['test_count'], 3); test.assertEqual(report['failures'], [])
                names = re.findall(rb'^Testing ([A-Za-z_][A-Za-z_0-9]*)\.\.\. ',out+b'\n'+err,re.M)
            else:
                names = re.findall(rb'^I am testing shadow ([A-Za-z_][A-Za-z_0-9]*)$',err,re.M)
            normalized = [re.sub(r'^__nano_module_+[0-9]+_', '', n.decode()) for n in names]
            test.assertEqual(Counter(normalized), Counter(['header_wrapper_0','header_wrapper_1','main']))
            test.command(name+'-run', [sys.executable, launch, empty, output])
            rows = [json.loads(line) for line in commands.read_text().splitlines()]
            if compiler == 'nanoc_c':
                module_rows = [row for row in rows if '-c' in row and any('.nano-module-' in arg and arg.endswith('/source.c') for arg in row)]
                test.assertGreaterEqual(len(module_rows), 2)
                # Each wrapper's actual generated source imports the leaf's
                # metadata header; the command must carry its declaring roots.
                for i in (0, 1):
                    matches = [row for row in module_rows if '-I'+str((work/f'wrapper {i}').resolve()) in row]
                    test.assertTrue(matches, (i, module_rows))
                    for row in matches:
                        test.assertEqual(row.count('-I'+str((work/f'leaf {i}').resolve())), 1)
                        test.assertEqual(row.count('-I'+str((work/f'include {i}').resolve())), 1)
            (work/(name+'-selection.json')).write_text(json.dumps(dict(
                names=normalized, commands=rows, outside_cwd=str(empty)),indent=2)+'\n')
    (work/'source-inputs.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in work.rglob('*') if p.is_file() and p.suffix in ('.nano','.h','.c')},indent=2)+'\n')
