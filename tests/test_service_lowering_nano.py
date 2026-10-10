"""I qualify independent Nano source lowering against the C source corpus."""
import os
from pathlib import Path
import re
import sys

from tests import test_service_lowering as corpus

ROOT = corpus.ROOT
CONSUMER = corpus.WRAPPER.replace(
    'extern NvmFileCyclicExecutionReport nvm_file_cyclic_program_source(NvmFileHostGrant *,const NvmFileCyclicOptions *,NvmFileScalar *);',
    '''#include <stdlib.h>
static unsigned char *bytes;
static size_t length;
static NvmFileCyclicExecutionReport nvm_file_cyclic_program_source(NvmFileHostGrant *grant,const NvmFileCyclicOptions *options,NvmFileScalar *scalar) {
    return nvm_file_execute_cyclic_bytes(grant,bytes,length,options,scalar);
}''').replace('int main(void) {', '''int main(int argc,char **argv) {
    assert(argc==2);
    FILE *file=fopen(argv[1],"rb");assert(file);
    assert(!fseek(file,0,SEEK_END));long size=ftell(file);assert(size>0);rewind(file);
    length=(size_t)size;bytes=malloc(length);assert(bytes);
    assert(fread(bytes,1,length,file)==length);assert(!fclose(file));''').replace(
    '    return 0;', '    free(bytes);\n    return 0;')


class ServiceLoweringNano(corpus.ServiceLowering):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.probes = []
        producers = [[ROOT/'bin/nano_virt']]
        driver = os.environ.get('NANO_SERVICE_LOWERING_DRIVER_MODULE')
        if driver:
            producers.append([ROOT/'bin/nano_vm', driver, '--'])
        for index, producer in enumerate(producers):
            print(f'I build independent File lowerer with producer {index}.', flush=True)
            module = cls.work/f'probe-{index}.nvm'
            cls.command([*producer, ROOT/'tests/service_lowering.nano', '--emit-nvm', '-o', module])
            generated = module.with_suffix('.c')
            native = module.with_suffix('.native')
            cls.command([ROOT/'bin/nvm2c', module, '-o', generated])
            cls.command([*cls.compiler, '-std=c11', '-O1', '-g', '-fsanitize=address,undefined',
                         '-fno-sanitize-recover=all', generated, '-o', native, '-lm',
                         *(['-ldl'] if sys.platform.startswith('linux') else [])])
            cls.probes.extend([[ROOT/'bin/nano_vm', module, '--'], [native]])
        consumer = cls.work/'consumer.c'
        consumer.write_text(CONSUMER)
        cls.consumer = cls.work/'consumer'
        cls.command([*cls.compiler, '-std=c11', '-O2', '-g', '-fsanitize=address,undefined',
                     '-fno-sanitize-recover=all', '-Isrc', consumer, ROOT/'lib/libnano_file_runtime.a',
                     '-o', cls.consumer, '-lm'])

    def lower(self, path, selection='main', expected=0, dependencies=()):
        outputs = []
        for command in self.probes:
            report = self.command([*command, path, selection, expected, *dependencies])
            self.assertIn(f'LOWER {expected} ', report)
            outputs.append(bytes(int(line[5:]) for line in report.splitlines() if line.startswith('BYTE ')))
        for other in outputs[1:]:
            self.assertEqual(outputs[0], other)
        if expected:
            self.assertEqual(outputs[0], b'')
        return outputs[0]

    def check(self, body, selection='main', expected=0, status=0, complete=False):
        path = self.work/'source.nano'
        path.write_text(body if complete else corpus.DECL+body)
        dependencies = []
        if complete and body.startswith('module "bridge.nano"'):
            dependencies = [self.work/'binding.nano', self.work/'bridge.nano']
        wire = self.lower(path, selection, dependencies=dependencies)
        module = self.work/'lowered.nvm'
        module.write_bytes(wire)
        vm = self.command([self.consumer, module])
        native_c = self.work/'source.c'
        native = self.work/'source.native'
        self.command([ROOT/'bin/nvm2c', '--file-temporary', '--file-cyclic', '--entry-name',
                      'source', module, '-o', native_c])
        self.command([*self.compiler, '-std=c11', '-O2', '-g', '-fsanitize=address,undefined',
                      '-fno-sanitize-recover=all', '-I'+str(self.work), '-Isrc', native_c,
                      self.work/'wrapper.c', ROOT/'lib/libnano_file_runtime.a', '-o', native, '-lm'])
        actual = self.command([native])
        pattern = r'EXEC (\d+) VALUE (-?\d+)'
        self.assertEqual(re.search(pattern, vm).groups(), re.search(pattern, actual).groups())
        self.assertEqual(re.search(pattern, actual).groups(), (str(status), str(expected)))

    def test_tcp_checked_source_stops_before_wire_publication(self):
        from tests.test_socket_service_source import source, tcp
        from tests.test_service_bodies import POSITIVE
        companion=self.work/'interface.nsi.json'
        previous=companion.read_bytes()
        try:
            companion.write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())
            path=self.work/'tcp.nano';path.write_text(source(tcp(POSITIVE)))
            output=self.work/'prior.c';output.write_bytes(b'prior-output')
            report=self.command([os.environ.get('NANO_SERVICE_LOWERING_RUNNER',ROOT/'obj/test_service_lowering'),path,'main',output,'lower:2'])
            self.assertIn('REFUSAL 2',report)
            self.assertEqual(output.read_bytes(),b'prior-output')
            self.lower(path,expected=2)
        finally:
            companion.write_bytes(previous)

    def test_limits_preserve_outputs(self):
        super().test_limits_preserve_outputs()
        cases = [
            ('fn main()->int{'+''.join(f'let n{i}:int={i} ' for i in range(257))+'return 0}', 3),
            (''.join(f'fn f{i}()->int{{return {i}}} ' for i in range(63))+'fn main()->int{return 0}', 3),
        ]
        for body, status in cases:
            path = self.work/'refused.nano'
            path.write_text(corpus.DECL+body)
            self.lower(path, expected=status)
