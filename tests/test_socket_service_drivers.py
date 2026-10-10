"""I qualify TCP source publication, selected shadows and invocation policy."""
import os
import shlex
import socket
import unittest
from pathlib import Path
from tests import test_service_drivers as file_drivers
ROOT = file_drivers.ROOT
from tests import test_socket_dispatch as socket_dispatch

class SocketServiceDrivers(unittest.TestCase):
    run_command = file_drivers.ServiceDrivers.run_command

    def setUp(self):
        file_drivers.ServiceDrivers.setUp(self)
        self.catalog.write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())
        self.env['NANOLANG_ROOT'] = str(ROOT)
        module = os.environ.get('NANO_TCP_DRIVER_MODULE')
        native = os.environ.get('NANO_TCP_DRIVER_NATIVE')
        for label, command in [('nano-driver-vm', [ROOT/'bin/nano_vm', module, '--']),
                               ('nano-driver-native', [native])]:
            if any(x is None for x in command):
                continue
            wrapper = self.work/label
            wrapper.write_text('#!/bin/sh\nexec '+shlex.join(list(map(str,command)))+' "$@"\n')
            wrapper.chmod(0o700)
            self.drivers.append(wrapper)

    def network_source(self, family):
        port, received = socket_dispatch.SocketDispatch.server(self, socket.AF_INET if family == 4 else socket.AF_INET6)
        endpoint = f'Endpoint {{ family: {family}, address0: {2130706433 if family == 4 else 0}, address1: 0, address2: 0, address3: {1 if family == 6 else 0}, port: {port}, scope_id: 0 }}'
        text = '''service "nsi:nanolang/net" catalog 1 from "interface.nsi.json"
fn exchange()->int {
 match (begin_connect ENDPOINT) {
  Error(e)=>{return -1}
  Ok(c)=>{
   let mut conn:Conn=c
   let mut ready:bool=false
   while (not ready) {match (finish_connect &mut conn) {Ok()=>{set ready true} Error(e)=>{assert (== e.status 2)}}}
   let mut sent:bool=false
   while (not sent) {match (send_byte &mut conn 165) {Ok(n)=>{assert (== n 1) set sent true} Error(e)=>{assert (== e.status 2)}}}
   let mut done:bool=false
   let mut value:int=0
   while (not done) {match (receive_byte &mut conn) {Ok(b)=>{assert (not b.eof) set value b.value set done true} Error(e)=>{assert (== e.status 2)}}}
   match (close conn) {Ok()=>{} Error(e)=>{assert false}}
   return value
  }
 }
}
shadow exchange {assert (== (exchange) 90)}
fn main()->int{return (exchange)}
'''.replace('ENDPOINT', endpoint)
        return text, received

    def test_network_shadow_publication_and_invocation(self):
        for family in (4, 6):
            text, received = self.network_source(family)
            self.source.write_text(text)
            outputs = []
            for driver in self.drivers:
                with self.subTest(family=family, driver=driver.name):
                    wire = self.work/(driver.name+f'-v{family}.nvm')
                    wire.write_bytes(b'prior-output')
                    for flags in ([], ['--allow-temporary-files']):
                        denied = self.run_command([driver, self.source, *flags, '--emit-nvm', '-o', wire], 1)
                        self.assertIn('--allow-tcp-connections', denied.stdout+denied.stderr)
                        self.assertEqual(wire.read_bytes(), b'prior-output')
                    result = self.run_command([driver, self.source, '--allow-tcp-connections', '--emit-nvm', '-o', wire])
                    records = result.stderr.splitlines()
                    selected = [x[7:] for x in records if x.startswith('SELECT ')]
                    self.assertEqual(len(selected), 1)
                    self.assertEqual(selected, [x[6:] for x in records if x.startswith('START ')])
                    self.assertEqual(selected, [x[5:] for x in records if x.startswith('DONE ')])
                    outputs.append(wire.read_bytes())
                    self.run_command([ROOT/'bin/nano_vm', wire], 1)
                    self.run_command([ROOT/'bin/nano_vm', '--allow-tcp-connections', '--socket-instruction-limit', '1000000', wire], 90)
                    self.run_command([ROOT/'bin/nano_vm', '--allow-tcp-connections', '--socket-instruction-limit', '0', wire], 1)
                    for flags in (['--allow-tcp-connections'], ['--socket-instruction-limit', '20'],
                                  ['--allow-tcp-connections', '--socket-instruction-limit', '1000001'],
                                  ['--allow-tcp-connections', '--socket-instruction-limit', '20', '--verify-only']):
                        self.run_command([ROOT/'bin/nano_vm', *flags, wire], 1)
                    emitted = self.work/'emitted.c'
                    self.run_command([ROOT/'bin/nvm2c', '--socket-tcp', '--entry-name', 'example', wire, '-o', emitted])
                    self.assertIn('nvm_socket_indirect_program_example', emitted.read_text())
                    native = self.work/(driver.name+f'-v{family}.native')
                    self.run_command([driver, self.source, '--allow-tcp-connections', '-o', native])
                    self.run_command([native], 1)
                    self.run_command([native, '--allow-temporary-files'], 1)
                    self.run_command([native, '--allow-tcp-connections'], 90)
            for output in outputs[1:]:
                self.assertEqual(output, outputs[0])
            self.assertTrue(received and all(x == b'\xa5' for x in received))
            self.run_command([ROOT/'bin/nano_virt', self.source, '--allow-tcp-connections', '--run'], 90)

    def test_failure_aliases_and_nonexecuting_emission(self):
        text, _ = self.network_source(4)
        for driver in self.drivers:
            output = self.work/'protected-output'
            self.source.write_text(text.replace('shadow exchange {assert (== (exchange) 90)}',
                                                'shadow exchange {let value:int=(exchange) assert false}'))
            output.write_bytes(b'prior-output')
            for flags in (['--emit-nvm'], []):
                self.run_command([driver, self.source, '--allow-tcp-connections', *flags, '-o', output], 1)
                self.assertEqual(output.read_bytes(), b'prior-output')
            self.source.write_text(text)
            for input_path in (self.source, self.catalog):
                before = input_path.read_bytes()
                self.run_command([driver, self.source, '--allow-tcp-connections', '--emit-nvm', '-o', input_path], 1)
                self.assertEqual(input_path.read_bytes(), before)
            # No selected shadows: emission creates no grant and execution still needs one.
            self.source.write_text(text.replace('shadow exchange {assert (== (exchange) 90)}', ''))
            self.run_command([driver, self.source, '--emit-nvm', '-o', output])
            damaged = self.work/'damaged.nvm'; damaged.write_bytes(output.read_bytes()[:-1])
            emitted = self.work/'prior.c'; emitted.write_bytes(b'prior-output')
            self.run_command([ROOT/'bin/nvm2c', '--socket-tcp', '--entry-name', 'example', damaged, '-o', emitted], 1)
            self.assertEqual(emitted.read_bytes(), b'prior-output')

    def test_dependency_shadow_selection_preserves_outputs(self):
        text, _ = self.network_source(4)
        binding = self.work/'binding.nano'
        binding.write_text(text.replace('fn exchange()', 'pub fn exchange()').replace('fn main()->int{return (exchange)}', '').replace('shadow exchange {assert (== (exchange) 90)}', 'shadow exchange {let value:int=(exchange) assert false}'))
        self.source.write_text('module "binding.nano" as network\nfn main()->int{return (network.exchange)}\nshadow main {assert true}\n')
        outputs = []
        for driver in self.drivers:
            output = self.work/(driver.name+'.imports.nvm'); output.write_bytes(b'prior-output')
            self.run_command([driver, self.source, '--allow-tcp-connections', '--emit-nvm', '-o', output], 1)
            self.assertEqual(output.read_bytes(), b'prior-output')
            result = self.run_command([driver, self.source, '--allow-tcp-connections', '--root-shadows-only', '--emit-nvm', '-o', output])
            self.assertEqual(sum(line.startswith('SELECT ') for line in result.stderr.splitlines()), 1)
            outputs.append(output.read_bytes())
        for wire in outputs[1:]:
            self.assertEqual(wire, outputs[0])
