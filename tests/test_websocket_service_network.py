"""I require real source lifecycles through compiler, VM and native products."""
import unittest
from tests import test_websocket_service_drivers as drivers
from tests import test_websocket_source_lowering as lowering
from tests import test_websocket_dispatch as dispatch

ROOT=drivers.ROOT

class WebSocketServiceNetwork(unittest.TestCase):
    setUp=drivers.WebSocketServiceDrivers.setUp
    run_command=drivers.WebSocketServiceDrivers.run_command
    server=dispatch.WebSocketDispatch.server

    def test_source_lifecycles_with_numeric_and_dns_hosts(self):
        self.bad_reply=False
        port,messages=self.server()
        count=0
        for host,indirect in [('127.0.0.1',False),('127.0.0.1',True),('localhost',False),('localhost',True)]:
            text=lowering.SOURCE.replace('ws://127.0.0.1:1/test',f'ws://{host}:{port}/?mode=test')
            if indirect:text=text.replace('let message:Message=(payload)','let factory:fn()->Message=payload let message:Message=(factory)')
            self.source.write_text(text)
            flags=['--allow-websocket-connections']
            if host=='localhost':flags+=['--allow-websocket-lookup','--websocket-resolver-helper',str(ROOT/'bin/nano-resolver')]
            wires=[]
            for driver in self.drivers:
                with self.subTest(host=host,indirect=indirect,driver=driver.name):
                    wire=self.work/(driver.name+'.nvm');native=self.work/(driver.name+'.native')
                    self.run_command([driver,self.source,*flags,'--emit-nvm','-o',wire])
                    wires.append(wire.read_bytes())
                    self.run_command([ROOT/'bin/nano_vm',*flags,'--websocket-instruction-limit','100000',wire],77)
                    self.run_command([driver,self.source,*flags,'-o',native])
                    self.run_command([native,*flags],77)
                    count+=4
            for other in wires[1:]:self.assertEqual(wires[0],other)
        self.assertEqual(messages,[b'a\0b']*count)
