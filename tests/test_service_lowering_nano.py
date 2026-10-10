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
        # I retain the C fixture's reference-map mutation checks used by the
        # inherited corpus, alongside independent Nano VM/native execution.
        reference = self.command([os.environ.get('NANO_SERVICE_LOWERING_RUNNER', ROOT/'obj/test_service_lowering'),
                                  path, selection, self.work/'c-reference.c', status])
        self.assertEqual(re.search(pattern, reference).groups(), re.search(pattern, vm).groups())
        return reference + vm

    def test_tcp_source_bytes_and_granted_execution(self):
        import socket
        from tests.test_socket_dispatch import SocketDispatch
        from tests.test_socket_service_source import source
        companion = self.work/'interface.nsi.json'
        previous = companion.read_bytes()
        (self.work/'nanolang/socket').symlink_to(ROOT/'src', target_is_directory=True)
        wrapper = corpus.WRAPPER.replace('file_cyclic', 'socket_indirect').replace('FileCyclic', 'SocketIndirect').replace('FileHost', 'SocketHost').replace('FileScalar', 'SocketScalar').replace('nvm_file_', 'nvm_socket_').replace('NVM_FILE_', 'NVM_SOCKET_').replace('temporary_files', 'tcp_connections')
        wrapper_path = self.work/'socket-wrapper.c'
        wrapper_path.write_text(wrapper)
        def check_source(body, selection, expected, label, status=0):
            path = self.work/f'{label}.nano'; path.write_text(source(body))
            generated = self.work/f'{label}.c'
            report = self.command([os.environ.get('NANO_SERVICE_LOWERING_RUNNER', ROOT/'obj/test_service_lowering'), path, selection, generated, status])
            wire = self.lower(path, selection)
            self.assertEqual(wire, Path(str(generated)+'.nvm').read_bytes())
            native = self.work/f'{label}.native'
            self.command([*self.compiler, '-std=c99', '-O2', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                          '-I'+str(self.work), '-Isrc', generated, wrapper_path, ROOT/'lib/libnano_socket_runtime.a', '-o', native, '-lm'])
            self.assertIn(f'EXEC {status} VALUE {expected}', report)
            self.assertIn(f'EXEC {status} VALUE {expected}', self.command([native]))

        try:
            companion.write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())
            for family in (4, 6):
                port, received = SocketDispatch.server(self, socket.AF_INET if family == 4 else socket.AF_INET6)
                endpoint = f'Endpoint {{ scope_id: 0, port: {port}, address3: {1 if family == 6 else 0}, address2: 0, address1: 0, address0: {2130706433 if family == 4 else 0}, family: {family} }}'
                body = r"""
fn pass(conn:Conn)->Conn {return conn}
fn send(conn:&mut Conn,octet:int)->SendResult {return (send_byte &mut conn octet)}
fn main()->int {
 match (begin_connect ENDPOINT) {
  Error(e)=>{return e.status}
  Ok(c)=>{
   let mut owned:Conn=(pass c)
   let mut ready:bool=false
   while (not ready) {match (finish_connect &mut owned) {Ok()=>{set ready true} Error(e)=>{assert (== e.status 2)}}}
   let mut sent:bool=false
   while (not sent) {match (send &mut owned 165) {Ok(n)=>{assert (== n 1) set sent true} Error(e)=>{assert (== e.status 2)}}}
   let mut read:bool=false
   let mut value:int=0
   while (not read) {match (receive_byte &mut owned) {Ok(b)=>{assert (not b.eof) set value b.value set read true} Error(e)=>{assert (== e.status 2)}}}
   match (close owned) {Ok()=>{} Error(e)=>{assert false}}
   return value
  }
 }
}
""".replace('ENDPOINT', endpoint)
                if family == 6:
                    body = body.replace('let mut owned:Conn=(pass c)', 'let identity:fn(Conn)->Conn=pass let writer:fn(&mut Conn,int)->SendResult=send let mut owned:Conn=(identity c)').replace('(send &mut owned 165)', '(writer &mut owned 165)')
                for selection in ('main', 'address'):
                    check_source(body, selection, 90 if selection == 'main' else 0, f'ipv{family}-{selection}')
                self.assertTrue(received)
                self.assertTrue(all(value == b'\xa5' for value in received))
            invalid = 'Endpoint { family: 4, address0: 2130706433, address1: 0, address2: 0, address3: 0, port: 0, scope_id: 0 }'
            check_source('fn main()->int {match (begin_connect '+invalid+') {Error(e)=>{return e.status} Ok(c)=>{let closed:CloseResult=(close c) return -1}}}', 'main', 4, 'invalid-endpoint')
            terminal = '(match (begin_connect '+invalid+') {Error(e)=>{return 17} Ok(c)=>{let closed:CloseResult=(close c) return 18}})'
            record = invalid.replace('port: 0', 'port: '+terminal)
            check_source('fn sink(first:int, endpoint:Endpoint)->int{return first} fn main()->int{return (sink 5 '+record+')}','main',17,'terminal-endpoint')
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
