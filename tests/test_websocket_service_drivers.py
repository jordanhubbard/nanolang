"""I qualify WebSocket source publication, shadows and explicit invocation policy."""
import os
from pathlib import Path
import shlex
import shutil
import unittest
from tests import test_service_drivers as base

ROOT=base.ROOT
DECL='service "nsi:nanolang/websocket" catalog 1 from "interface.nsi.json"\n'
MESSAGE='''fn payload()->Message {return Message {data:"a\\0b",binary:true}}
shadow payload {let value:Message=(payload) assert value.binary assert (== value.data "a\\0b")}
'''
MAIN='fn main()->int {let value:Message=(payload) assert value.binary return 7}\n'
LOOKUP='''fn probe()->int {
 match (connect "ws://localhost:1/test" 1000) {
  Error(e)=>{return e.status}
  Ok(connection)=>{let result:CloseResult=(close connection 1000) match result {Ok()=>{return 77} Error(e)=>{return e.status}}}
 }
}
shadow probe {assert (== (probe) 2)}
fn main()->int{return (probe)}
'''

class WebSocketServiceDrivers(unittest.TestCase):
    run_command=base.ServiceDrivers.run_command

    def setUp(self):
        base.ServiceDrivers.setUp(self)
        self.catalog.write_bytes((ROOT/'tests/fixtures/nsi_websocket_plan.json').read_bytes())
        self.env['NANOLANG_ROOT']=str(ROOT)
        module=os.environ.get('NANO_WEBSOCKET_DRIVER_MODULE')
        if module:
            wrapper=self.work/'nano_driver_vm'
            wrapper.write_text('#!/bin/sh\nexec '+shlex.join([str(ROOT/'bin/nano_vm'),module,'--'])+' "$@"\n')
            wrapper.chmod(0o700);self.drivers.append(wrapper)
        native=os.environ.get("NANO_WEBSOCKET_DRIVER_NATIVE")
        if native:self.drivers.append(Path(native))

    def test_source_shadows_and_invocation_policy(self):
        self.source.write_text(DECL+MESSAGE+MAIN)
        modules=[]
        for driver in self.drivers:
            with self.subTest(driver=driver.name):
                wire=self.work/(driver.name+'.nvm')
                report=self.run_command([driver,self.source,'--allow-websocket-connections','--emit-nvm','-o',wire])
                self.assertEqual(sum(x.startswith('DONE ') for x in report.stderr.splitlines()),1)
                modules.append(wire.read_bytes())
                self.run_command([ROOT/'bin/nano_vm','--allow-websocket-connections','--websocket-instruction-limit','100000',wire],7)
                self.run_command([ROOT/'bin/nano_vm','--allow-websocket-connections','--websocket-instruction-limit','0',wire],1)
                for flags in [[],['--allow-websocket-connections'],['--websocket-instruction-limit','100'],
                              ['--allow-websocket-connections','--websocket-instruction-limit','1000001'],
                              ['--allow-websocket-connections','--websocket-instruction-limit','100','--services'],
                              ['--allow-websocket-connections','--websocket-instruction-limit','100','--allow-websocket-lookup']]:
                    self.run_command([ROOT/'bin/nano_vm',*flags,wire],1)
                if driver.name=='nano_virt':self.run_command([driver,self.source,'--allow-websocket-connections','--run'],7)
                native=self.work/(driver.name+'.native')
                self.run_command([driver,self.source,'--allow-websocket-connections','-o',native])
                self.run_command([native],1)
                self.run_command([native,'--allow-tcp-connections'],1)
                self.run_command([native,'--allow-websocket-connections'],7)
                self.run_command([native,'--allow-websocket-connections','--allow-websocket-connections'],1)
                self.run_command([native,'--allow-websocket-connections','--allow-websocket-lookup'],1)
                emitted=self.work/(driver.name+'.c')
                self.run_command([ROOT/'bin/nvm2c','--websocket','--entry-name','source',wire,'-o',emitted])
                self.assertIn('nvm_websocket_indirect_program_source',emitted.read_text())
        for other in modules[1:]:self.assertEqual(modules[0],other)

    def test_lookup_requires_separate_authority(self):
        self.source.write_text(DECL+LOOKUP)
        for driver in self.drivers:
            with self.subTest(driver=driver.name):
                wire=self.work/(driver.name+'.nvm');native=self.work/(driver.name+'.native')
                self.run_command([driver,self.source,'--allow-websocket-connections','--emit-nvm','-o',wire])
                self.run_command([ROOT/'bin/nano_vm','--allow-websocket-connections','--websocket-instruction-limit','100000',wire],2)
                self.run_command([driver,self.source,'--allow-websocket-connections','-o',native])
                self.run_command([native,'--allow-websocket-connections'],2)

    def test_refusals_preserve_prior_outputs(self):
        self.source.write_text(DECL+MESSAGE+MAIN)
        for driver in self.drivers:
            for flags in [[],['--allow-tcp-connections'],['--allow-temporary-files'],
                          ['--allow-websocket-connections','--allow-websocket-lookup'],
                          ['--allow-websocket-connections','--allow-websocket-lookup','--websocket-resolver-helper','relative'],
                          ['--allow-websocket-connections','--websocket-resolver-helper','/unused'],
                          ['--allow-websocket-connections','--websocket-resolver-helper','']]:
                with self.subTest(driver=driver.name,flags=flags):
                    output=self.work/'prior.nvm';output.write_bytes(b'prior')
                    self.run_command([driver,self.source,*flags,'--emit-nvm','-o',output],1)
                    self.assertEqual(output.read_bytes(),b'prior')
            self.source.write_text(DECL+MESSAGE.replace('assert value.binary','assert false')+MAIN)
            output=self.work/'prior.nvm';output.write_bytes(b'prior')
            self.run_command([driver,self.source,'--allow-websocket-connections','--emit-nvm','-o',output],1)
            self.assertEqual(output.read_bytes(),b'prior')
            self.source.write_text(DECL+MESSAGE+MAIN)

    def test_installed_and_relocated_native_publication(self):
        self.source.write_text(DECL+MESSAGE+MAIN)
        prefix=self.work/'installed'
        self.run_command(['make','-f','Makefile.gnu','-C',ROOT,'PREFIX='+str(prefix),'install-websocket-public-runtime'])
        (prefix/'bin').mkdir()
        for name in ['nanoc_c','nano_virt']:
            shutil.copy2(ROOT/'bin'/name,prefix/'bin'/name)
        moved=self.work/'relocated';prefix.rename(moved)
        env=dict(self.env);env.pop('NANO_ROOT',None)
        for name in ['nanoc_c','nano_virt']:
            native=self.work/(name+'.installed')
            self.run_command([moved/'bin'/name,self.source,'--allow-websocket-connections','-o',native],env=env)
            self.run_command([native,'--allow-websocket-connections'],7)

    def test_nonexecuting_emission_and_source_aliases(self):
        self.source.write_text(DECL+'fn main()->int{return 7}\n')
        wires=[]
        for driver in self.drivers:
            output=self.work/(driver.name+'.nvm')
            self.run_command([driver,self.source,'--emit-nvm','-o',output])
            wires.append(output.read_bytes())
            self.run_command([ROOT/'bin/nano_vm','--allow-websocket-connections','--websocket-instruction-limit','100',output],7)
            for protected in [self.source,self.catalog]:
                previous=protected.read_bytes()
                self.run_command([driver,self.source,'--emit-nvm','-o',protected],1)
                self.assertEqual(previous,protected.read_bytes())
            generated=self.work/'prior.c';generated.write_bytes(b'prior')
            self.run_command([ROOT/'bin/nvm2c','--socket-tcp','--entry-name','wrong',output,'-o',generated],1)
            self.assertEqual(generated.read_bytes(),b'prior')
        for other in wires[1:]:self.assertEqual(wires[0],other)
