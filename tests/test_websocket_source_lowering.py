"""I compare checked C/Nano WebSocket wire and execute explicit denied grants."""
import shlex
from pathlib import Path
import sys
import unittest
from tests import test_service_lowering as base
from tests import test_service_lowering_nano as nano

ROOT = base.ROOT
SOURCE = '''service "nsi:nanolang/websocket" catalog 1 from "interface.nsi.json"
fn payload()->Message {return Message {data:"a\\0b",binary:true}}
shadow payload {let value:Message=(payload) assert value.binary assert (== value.data "a\\0b")}
fn emit(value:&mut Connection,message:Message)->SendResult {return (send &mut value message 1000)}
fn main()->int {
 let message:Message=(payload)
 assert message.binary assert (== message.data "a\\0b")
 let result:ConnectResult=(connect "ws://127.0.0.1:1/test" 1000)
 match result {
  Ok(conn)=>{let mut owned:Connection=conn let mut i:int=0
   while (< i 2) {let sent:SendResult=(emit &mut owned message)
    match sent {Ok(n)=>{assert (== n 3)} Error(e)=>{assert false}}
    let received:ReceiveResult=(receive &mut owned 1000)
    match received {Ok(value)=>{assert value.binary assert (== value.data message.data)} Error(e)=>{assert false}}
    set i (+ i 1)
   }
   let closed:CloseResult=(close owned 1000)
   match closed {Ok()=>{return 77} Error(e)=>{return e.status}}
  }
  Error(e)=>{return e.status}
 }
}
'''
WRAPPER = '''#include <nanolang/websocket/nanoisa/websocket_indirect_public.h>
#include <assert.h>
#include <stdio.h>
extern NvmWebSocketIndirectExecutionReport nvm_websocket_indirect_program_source(NvmWebSocketHostGrant *,const NvmWebSocketIndirectOptions *,NvmWebSocketScalar *);
int main(void){NvmWebSocketHostPolicy policy={1,false,false,2000,NULL};NvmWebSocketHostGrant *grant=NULL;
 assert(nvm_websocket_host_grant_create(&policy,&grant)==NVM_WEBSOCKET_HOST_OK);
 NvmWebSocketIndirectOptions options={1,100000};NvmWebSocketScalar scalar={0};
 NvmWebSocketIndirectExecutionReport report=nvm_websocket_indirect_program_source(grant,&options,&scalar);
 assert(report.runtime.status==0 && !report.runtime.cleanup.cleanup_failures);
 printf("EXEC %u VALUE %lld\\n",report.runtime.status,(long long)scalar.value);
 assert(nvm_websocket_host_grant_destroy(&grant)==NVM_WEBSOCKET_HOST_OK);return 0;}
'''

class WebSocketSourceLowering(unittest.TestCase):
    command = staticmethod(base.ServiceLowering.command)
    lower = nano.ServiceLoweringNano.lower

    @classmethod
    def setUpClass(cls):
        base.ServiceLowering.setUpClass.__func__(cls)
        (cls.work/'interface.nsi.json').write_bytes((ROOT/'tests/fixtures/nsi_websocket_plan.json').read_bytes())
        (cls.work/'nanolang/websocket').symlink_to(ROOT/'src',target_is_directory=True)
        (cls.work/'wrapper.c').write_text(WRAPPER)
        cls.probes=[]
        module=cls.work/'lowerer.nvm'
        cls.command([ROOT/'bin/nano_virt',ROOT/'tests/service_lowering.nano','--emit-nvm','-o',module])
        generated=module.with_suffix('.c');native=module.with_suffix('.native')
        cls.command([ROOT/'bin/nvm2c',module,'-o',generated])
        cls.command([*cls.compiler,'-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            generated,'-o',native,'-lm',*(['-ldl'] if sys.platform.startswith('linux') else [])])
        cls.probes=[[ROOT/'bin/nano_vm',module,'--'],[native]]

    def test_exact_wire_and_vm_native_results(self):
        for indirect in (False,True):
            text=SOURCE
            if indirect:
                text=text.replace('let message:Message=(payload)','let factory:fn()->Message=payload let message:Message=(factory)')
            path=self.work/'source.nano';path.write_text(text)
            for selection,expected in [('main',2),('payload',0)]:
                with self.subTest(indirect=indirect,selection=selection):
                    generated=self.work/'source.c'
                    actual=self.command([ROOT/'obj/test_service_lowering',path,selection,generated])
                    self.assertIn(f'EXEC 0 VALUE {expected}',actual)
                    wire=self.lower(path,selection)
                    self.assertEqual(wire,Path(str(generated)+'.nvm').read_bytes())
                    native=self.work/'source.native'
                    self.command([*self.compiler,'-std=c11','-O2','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                        '-I'+str(self.work),generated,self.work/'wrapper.c',ROOT/'lib/libnano_websocket_runtime.a',
                        *shlex.split(self.command(['pkg-config','--libs','libcrypto'])),'-lm','-o',native])
                    self.assertIn(f'EXEC 0 VALUE {expected}',self.command([native]))

    def test_preserved_tcp_record_and_wire(self):
        from tests.test_socket_service_source import source
        companion=self.work/'interface.nsi.json'
        previous=companion.read_bytes()
        try:
            companion.write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())
            text=source('fn main()->int {let endpoint:Endpoint=(address) return endpoint.port}')
            text=text.replace('family: 4, address0: 2130706433, address1: 0, address2: 0, address3: 0, port: 12345, scope_id: 0',
                'scope_id: 0, port: 12345, address3: 0, address2: 0, address1: 0, address0: 2130706433, family: 4')
            path=self.work/'tcp.nano';path.write_text(text)
            generated=self.work/'tcp.c'
            self.assertIn('EXEC 0 VALUE 12345',self.command([ROOT/'obj/test_service_lowering',path,'main',generated]))
            self.assertEqual(self.lower(path),Path(str(generated)+'.nvm').read_bytes())
        finally:
            companion.write_bytes(previous)
