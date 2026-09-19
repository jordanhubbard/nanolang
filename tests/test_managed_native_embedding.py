"""I qualify standalone source embedding without selecting mixed bytecode."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('managed_strings.h', 'binary64_parse.h', 'managed_strings.c')


def source_bytes():
    data = [(ROOT / 'src/nanoisa' / name).read_bytes() for name in SOURCES]
    runtime = data[2]
    for name in SOURCES[:2]:
        runtime = runtime.replace(f'#include "{name}"\n'.encode(), b'', 1)
    return data[0] + data[1] + runtime


def embedded_bytes(path):
    return ''.join(ast.literal_eval(line) for line in path.read_text().splitlines()
                   if line.startswith('"')).encode('latin1')


HARNESS = r'''
#include <assert.h>
int main(void) {
    static const NmsRecordDescriptor descriptors[]={{7,1}};
    NmsRuntime runtime;
    nms_init(&runtime,NULL,0);
    assert(nms_bind_records(&runtime,descriptors,1)==NMS_OK);
    assert(nms_begin(&runtime)==NMS_OK);
    NmsHandle array=0,record=0;
    assert(nms_vm_array_create(&runtime,3,&array)==NMS_OK);
    NmsValue one={UINT64_C(0x3ff8000000000000),3};
    assert(nms_value_array_append(&runtime,array,one)==NMS_OK);
    NmsValue child={array,NMS_ARRAY_TAG};
    assert(nms_record_create(&runtime,0,&child,1,&record)==NMS_OK);
    assert(nms_release(&runtime,array)==NMS_OK);
    uint32_t ordinal=99,global=99;
    assert(nms_record_identity(&runtime,record,&ordinal,&global)==NMS_OK);
    assert(ordinal==0 && global==7);
    NmsValue alias={0,0};
    assert(nms_record_get(&runtime,record,0,&alias)==NMS_OK);
    assert(nms_release(&runtime,record)==NMS_OK);
    NmsValue got={0,0};
    assert(nms_value_array_get(&runtime,alias.payload,0,&got)==NMS_OK);
    assert(got.tag==3 && got.payload==one.payload);
    assert(nms_value_release(&runtime,got)==NMS_OK);
    assert(nms_value_array_get(&runtime,alias.payload,UINT64_MAX,&got)==NMS_OK);
    assert(got.tag==0);
    assert(nms_value_release(&runtime,alias)==NMS_OK);
    /* I check real roots before terminal disposal can sweep them. */
    assert(runtime.live_objects==0 && runtime.live_bytes==0);
    assert(nms_finish(&runtime,NMS_OK,0)==0);
    assert(!runtime.active);
    assert(nms_dispose(&runtime)==NMS_OK);
    assert(nano_rt_f64_add(1.5,2.5)==4.0);
    assert(nano_rt_f64_sub(4.0,2.5)==1.5);
    assert(nano_rt_f64_mul(1.5,2.0)==3.0);
    assert(nano_rt_f64_div(3.0,2.0)==1.5);
    return 0;
}
'''


class ManagedNativeEmbedding(unittest.TestCase):
    def test_exact_bytes_and_hashes(self):
        header = ROOT / 'src/nanoisa/managed_native_source.h'
        expected = source_bytes()
        self.assertEqual(embedded_bytes(header), expected)
        self.assertIn(hashlib.sha256(expected).hexdigest(), header.read_text())
        for name in SOURCES:
            self.assertIn(hashlib.sha256((ROOT / 'src/nanoisa' / name).read_bytes()).hexdigest(),
                          header.read_text())
        subprocess.run(['python3', 'scripts/embed_managed_native.py', '--check'],
                       cwd=ROOT, check=True, timeout=20)

    def test_isolated_regeneration_and_refusals(self):
        with tempfile.TemporaryDirectory(prefix='nms-embed-generator-') as temporary:
            root = Path(temporary)
            (root / 'scripts').mkdir()
            directory = root / 'src/nanoisa'
            directory.mkdir(parents=True)
            shutil.copy2(ROOT / 'scripts/embed_managed_native.py', root / 'scripts')
            for name in SOURCES:
                shutil.copy2(ROOT / 'src/nanoisa' / name, directory)
            command = ['python3', str(root / 'scripts/embed_managed_native.py')]
            def run(*args):
                return subprocess.run(command + list(args), capture_output=True, timeout=20)
            self.assertNotEqual(run('--check').returncode, 0)
            self.assertEqual(run().returncode, 0)
            output = directory / 'managed_native_source.h'
            original = output.read_bytes()
            self.assertEqual(original, (ROOT / 'src/nanoisa/managed_native_source.h').read_bytes())
            self.assertEqual(run('--check').returncode, 0)
            self.assertEqual(run().returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            output.write_bytes(b'stale')
            self.assertNotEqual(run('--check').returncode, 0)
            self.assertEqual(run().returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            runtime = directory / 'managed_strings.c'
            runtime.write_bytes(runtime.read_bytes() + b'#include "unrecorded.h"\n')
            self.assertNotEqual(run().returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            shutil.copy2(ROOT / 'src/nanoisa/managed_strings.c', runtime)
            runtime.write_bytes(runtime.read_bytes() + b'#include "managed_strings.h"\n')
            self.assertNotEqual(run().returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_strict_standalone_c_and_existing_helpers(self):
        evidence = os.environ.get('NMS_EMBED_EVIDENCE')
        temporary = None if evidence else tempfile.TemporaryDirectory(prefix='nms-native-embed-')
        directory = Path(evidence or temporary.name)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            # The existing parser precedes the embedded duplicate: its normal
            # guard must prevent duplicate definitions. Arithmetic follows.
            source = (embedded_bytes(ROOT / 'src/nanoisa/binary64_parse_source.h') +
                      embedded_bytes(ROOT / 'src/binary64_arithmetic_source.h') +
                      embedded_bytes(ROOT / 'src/nanoisa/managed_native_source.h') + HARNESS.encode())
            path = directory / 'standalone.c'
            path.write_bytes(source)
            for compiler in (os.environ.get('NMS_EMBED_GCC', 'gcc'),
                             os.environ.get('NMS_EMBED_CLANG', 'clang')):
                for optimization in ('-O0', '-O2'):
                    name = ('clang' if compiler == os.environ.get('NMS_EMBED_CLANG', 'clang') else 'gcc') + optimization[1:]
                    with self.subTest(compiler=compiler, optimization=optimization):
                        exe = directory / name
                        command = [compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                                   '-pedantic', optimization, str(path), '-o', str(exe)]
                        built = subprocess.run(command, capture_output=True, timeout=60)
                        (directory / (name + '.build.log')).write_bytes(built.stdout + built.stderr)
                        self.assertEqual(built.returncode, 0, built.stderr.decode())
                        result = subprocess.run([str(exe)], capture_output=True, timeout=20)
                        (directory / (name + '.run.log')).write_bytes(result.stdout + result.stderr)
                        self.assertEqual(result.returncode, 0, result.stderr.decode())
                        dependency_command = (['otool', '-L', str(exe)] if sys.platform == 'darwin'
                                              else ['readelf', '-d', str(exe)])
                        dependencies = subprocess.run(dependency_command, capture_output=True, timeout=20)
                        self.assertEqual(dependencies.returncode, 0)
                        (directory / (name + '.dynamic.log')).write_bytes(dependencies.stdout)
                        if sys.platform != 'darwin':
                            self.assertNotIn(b'libm.so', dependencies.stdout)
                        report = dict(command=command, build_status=built.returncode,
                                      run_status=result.returncode, dependency_command=dependency_command,
                                      source_sha256=hashlib.sha256(source).hexdigest(),
                                      executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
                        (directory / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        finally:
            if temporary:
                temporary.cleanup()
