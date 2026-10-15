"""I qualify mixed source publication, independent grants and selected shadows."""
import os
import re
import socket
import subprocess
import unittest
from pathlib import Path
from tests import test_socket_service_drivers as tcp_drivers
from tests import test_socket_dispatch as socket_dispatch
ROOT=tcp_drivers.ROOT
FLAGS=['--allow-temporary-files','--allow-tcp-connections']

class MixedServiceDrivers(unittest.TestCase):
    run_command=tcp_drivers.SocketServiceDrivers.run_command
    setUp=tcp_drivers.SocketServiceDrivers.setUp

    def graph(self,family=4):
        self.catalog.write_bytes((ROOT/'tests/fixtures/nsi_file_plan.json').read_bytes())
        tcp_catalog=self.work/'tcp.nsi.json'
        tcp_catalog.write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())
        port,received=socket_dispatch.SocketDispatch.server(self,socket.AF_INET if family==4 else socket.AF_INET6)
        files='''service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"
pub fn pass(value:File)->File{return value}
pub fn write(value:&mut File,n:int)->WriteResult{return (write_byte &mut value n)}
pub fn read(value:&mut File,n:int)->void {
 match (rewind &mut value) {Ok()=>{} Error(e)=>{assert false}}
 match (read_byte &mut value) {Ok(b)=>{assert (== b.value n)} Error(e)=>{assert false}}
}
pub fn smoke()->int {match (temp) {Error(e)=>{return -1} Ok(f)=>{let c:CloseResult=(close f) return 0}}}
shadow smoke {assert (== (smoke) 0)}
'''
        first=self.work/'first.nano';first.write_text(files)
        tcp=self.work/'tcp.nano';tcp.write_text('''service "nsi:nanolang/net" catalog 1 from "tcp.nsi.json"
pub fn pass(value:Conn)->Conn{return value}
pub fn exchange(conn:&mut Conn)->int {
 let mut ready:bool=false
 while (not ready) {match (finish_connect &mut conn) {Ok()=>{set ready true} Error(e)=>{assert (== e.status 2)}}}
 let mut sent:bool=false
 while (not sent) {match (send_byte &mut conn 165) {Ok(n)=>{assert (== n 1) set sent true} Error(e)=>{assert (== e.status 2)}}}
 let mut read:bool=false let mut value:int=0
 while (not read) {match (receive_byte &mut conn) {Ok(b)=>{assert (not b.eof) set value b.value set read true} Error(e)=>{assert (== e.status 2)}}}
 return value
}
pub fn marker()->int{return 0}
shadow marker {assert (== (marker) 0)}
''')
        second=self.work/'second.nano';second.write_text(files)
        endpoint=f'tcp.Endpoint {{ scope_id: 0, port: {port}, address3: {1 if family==6 else 0}, address2: 0, address1: 0, address0: {2130706433 if family==4 else 0}, family: {family} }}'
        self.source.write_text('''module "first.nano" as first
module "tcp.nano" as tcp
module "second.nano" as second
fn run()->int {
 match (first.temp) {Error(e)=>{return e.status} Ok(a)=>{
  let identity:fn(first.File)->first.File=first.pass
  let writer:fn(&mut first.File,int)->first.WriteResult=first.write
  let mut left:first.File=(identity a)
  match (tcp.begin_connect ENDPOINT) {Error(e)=>{let c:first.CloseResult=(first.close left) return e.status} Ok(c)=>{
   let mut conn:tcp.Conn=(tcp.pass c)
   match (second.temp) {Error(e)=>{let l:first.CloseResult=(first.close left) let c:tcp.CloseResult=(tcp.close conn) return e.status} Ok(b)=>{
    let mut right:second.File=(second.pass b)
    match (writer &mut left 17) {Ok(n)=>{assert (== n 1)} Error(e)=>{assert false}}
    match (second.write &mut right 23) {Ok(n)=>{assert (== n 1)} Error(e)=>{assert false}}
    (first.read &mut left 17) (second.read &mut right 23)
    let value:int=(tcp.exchange &mut conn)
    match (first.close left) {Ok()=>{} Error(e)=>{assert false}}
    match (tcp.close conn) {Ok()=>{} Error(e)=>{assert false}}
    match (second.close right) {Ok()=>{} Error(e)=>{assert false}}
    return value
   }}
  }}
 }}
}
shadow run {assert (== (run) 90)}
fn main()->int{return (run)}
'''.replace('ENDPOINT',endpoint))
        self.inputs=[self.source,first,tcp,second,self.catalog,tcp_catalog]
        return received

    def records(self,run,count):
        lines=run.stderr.splitlines()
        selected=[x[7:] for x in lines if x.startswith('SELECT ')]
        self.assertEqual(len(selected),count,run.stderr)
        self.assertEqual(selected,[x[6:] for x in lines if x.startswith('START ')])
        self.assertEqual(selected,[x[5:] for x in lines if x.startswith('DONE ')])

    def test_network_publication_and_raw_routes(self):
        for family in (4,6):
            received=self.graph(family);outputs=[]
            for driver in self.drivers:
                wire=self.work/(driver.name+f'-v{family}.nvm');wire.write_bytes(b'prior-output')
                for flags in ([],FLAGS[:1],FLAGS[1:]):
                    result=self.run_command([driver,self.source,*flags,'--emit-nvm','-o',wire],1)
                    self.assertIn('--allow-',result.stdout+result.stderr)
                    self.assertEqual(wire.read_bytes(),b'prior-output')
                result=self.run_command([driver,self.source,*FLAGS,'--emit-nvm','-o',wire]);self.records(result,4)
                outputs.append(wire.read_bytes())
                native=self.work/(driver.name+f'-v{family}.native')
                result=self.run_command([driver,self.source,*FLAGS,'-o',native]);self.records(result,4)
                for flags in ([],FLAGS[:1],FLAGS[1:]):self.run_command([native,*flags],1)
                self.run_command([native,*FLAGS],90)
                self.run_command([native,*reversed(FLAGS)],90)
                symbols=subprocess.run(['nm',str(native)],capture_output=True,text=True,check=True).stdout
                self.assertNotRegex(symbols,r'\b_?(vm_execute|nvm_services_execute_indirect_bytes|nvm2c_emit_services_indirect_bytes)\b')
            for wire in outputs[1:]:self.assertEqual(outputs[0],wire)
            raw=self.work/'raw.nvm';raw.write_bytes(outputs[0])
            route=[ROOT/'bin/nano_vm','--services','--service-instruction-limit','1000000']
            for flags in ([],FLAGS[:1],FLAGS[1:]):self.run_command([*route,*flags,raw],1)
            self.run_command([*route,*FLAGS,raw],90)
            self.run_command([ROOT/'bin/nano_vm',raw],1)
            self.run_command([ROOT/'bin/nano_vm','--services','--service-instruction-limit','0',*FLAGS,raw],1)
            for extra in (['--verify-only'],['--file-indirect'],['--socket-instruction-limit','10'],['--services'],['--service-instruction-limit','10'],['--','guest']):
                self.run_command([*route,*FLAGS,raw,*extra],1)
            self.run_command([ROOT/'bin/nano_vm','--services',*FLAGS,raw],1)
            self.run_command([ROOT/'bin/nano_vm','--services','--service-instruction-limit','1000001',*FLAGS,raw],1)
            emitted=self.work/'emitted.c'
            self.run_command([ROOT/'bin/nvm2c','--services','--entry-name','example',raw,'-o',emitted])
            self.assertIn('nvm_services_indirect_program_example',emitted.read_text())
            emitted.write_bytes(b'prior-output')
            self.run_command([ROOT/'bin/nvm2c','--services','--socket-tcp','--entry-name','example',raw,'-o',emitted],2)
            self.assertEqual(emitted.read_bytes(),b'prior-output')
            self.run_command([ROOT/'bin/nano_virt',self.source,*FLAGS,'--run'],90)
            self.assertTrue(received and all(value==b'\xa5' for value in received))

    def test_shadow_failure_aliases_and_nonexecuting_publication(self):
        self.graph();original={p:p.read_bytes() for p in self.inputs}
        for driver in self.drivers:
            output=self.work/'protected-output'
            self.source.write_bytes(original[self.source].replace(b'shadow run {assert (== (run) 90)}',b'shadow run {let n:int=(run) assert false}'))
            output.write_bytes(b'prior-output')
            for flags in (['--emit-nvm'],[]):
                self.run_command([driver,self.source,*FLAGS,*flags,'-o',output],1)
                self.assertEqual(output.read_bytes(),b'prior-output')
            self.source.write_bytes(original[self.source])
            for p in self.inputs:
                before=p.read_bytes()
                self.run_command([driver,self.source,*FLAGS,'--emit-nvm','-o',p],1)
                self.assertEqual(p.read_bytes(),before)
            for p in self.inputs[:4]:p.write_text(re.sub(r'^shadow .*\n','',original[p].decode(),flags=re.MULTILINE))
            self.run_command([driver,self.source,'--emit-nvm','-o',output])
            broken=self.work/'broken.nvm';broken.write_bytes(output.read_bytes()[:-1])
            emitted=self.work/'prior.c';emitted.write_bytes(b'prior-output')
            self.run_command([ROOT/'bin/nvm2c','--services','--entry-name','example',broken,'-o',emitted],1)
            self.assertEqual(emitted.read_bytes(),b'prior-output')
            native=self.work/'emit-only-native';self.run_command([driver,self.source,'-o',native])
            self.run_command([native],1);self.run_command([native,*FLAGS],90)
            for p,value in original.items():p.write_bytes(value)

    def test_dependency_shadow_selection(self):
        self.graph();first=self.work/'first.nano'
        first.write_text(first.read_text().replace('shadow smoke {assert (== (smoke) 0)}','shadow smoke {let n:int=(smoke) assert false}'))
        outputs=[]
        for driver in self.drivers:
            output=self.work/(driver.name+'.root.nvm');output.write_bytes(b'prior-output')
            self.run_command([driver,self.source,*FLAGS,'--emit-nvm','-o',output],1)
            self.assertEqual(output.read_bytes(),b'prior-output')
            result=self.run_command([driver,self.source,*FLAGS,'--root-shadows-only','--emit-nvm','-o',output])
            self.records(result,1);outputs.append(output.read_bytes())
        for wire in outputs[1:]:self.assertEqual(outputs[0],wire)

    def test_repeated_single_catalog_permissions(self):
        for catalog,flag in ((1,FLAGS[0]),(2,FLAGS[1])):
            self.catalog.write_bytes((ROOT/('tests/fixtures/nsi_file_plan.json' if catalog==1 else 'tests/fixtures/nsi_socket_plan.json')).read_bytes())
            interface='filesystem' if catalog==1 else 'net'
            for name in ('first','second'):
                (self.work/(name+'.nano')).write_text(f'service "nsi:nanolang/{interface}" catalog 1 from "interface.nsi.json"\n')
            self.source.write_text('module "first.nano" as first\nmodule "second.nano" as second\nfn main()->int{return 7}\nshadow main {assert (== (main) 7)}\n')
            modules=[]
            for driver in self.drivers:
                wire=self.work/(driver.name+f'-catalog{catalog}.nvm');wire.write_bytes(b'prior-output')
                other=FLAGS[1] if catalog==1 else FLAGS[0]
                self.run_command([driver,self.source,other,'--emit-nvm','-o',wire],1)
                self.assertEqual(wire.read_bytes(),b'prior-output')
                self.run_command([driver,self.source,flag,'--emit-nvm','-o',wire])
                modules.append(wire.read_bytes())
                self.run_command([ROOT/'bin/nano_vm','--services','--service-instruction-limit','100000',flag,wire],7)
                native=self.work/(driver.name+f'-catalog{catalog}.native')
                self.run_command([driver,self.source,flag,'-o',native])
                self.run_command([native,other],1);self.run_command([native,flag],7)
            for wire in modules[1:]:self.assertEqual(modules[0],wire)

    def test_installed_runtime_publication(self):
        prefix=os.environ.get('NANO_MIXED_INSTALL_PREFIX')
        if not prefix:self.skipTest('I require an installed mixed runtime prefix for this package gate')
        prefix=Path(prefix);self.assertFalse((prefix/'src').exists())
        self.graph();env={**self.env,'NANO_ROOT':str(prefix)}
        for driver in self.drivers[:2]:
            native=self.work/(driver.name+'.installed')
            result=self.run_command([driver,self.source,*FLAGS,'-o',native],env=env);self.records(result,4)
            self.run_command([native],1);self.run_command([native,*FLAGS],90)
