"""I qualify paired mixed WebSocket source bytes and product permissions."""
from pathlib import Path
import shutil
import shlex
import unittest
from tests import test_websocket_service_drivers as drivers
from tests import test_websocket_source_lowering as lowering
from tests import test_websocket_dispatch as dispatch
from tests import test_service_lowering as base
from tests import test_service_lowering_nano as nano

ROOT=base.ROOT
WS='--allow-websocket-connections'
FLAGS=[WS,'--allow-temporary-files','--allow-tcp-connections']


def graph(work,catalogs=(3,1,2,3),url='ws://localhost:1/test',expected=2):
    names={1:('filesystem','file'),2:('net','socket'),3:('websocket','websocket')}
    dependencies=[];imports=[];calls=[];flags=[]
    for index,catalog in enumerate(catalogs):
        interface,fixture=names[catalog]
        companion=work/f'catalog{index}.json'
        companion.write_bytes((ROOT/f'tests/fixtures/nsi_{fixture}_plan.json').read_bytes())
        declaration=f'service "nsi:nanolang/{interface}" catalog 1 from "{companion.name}"\n'
        path=work/f'part{index}.nano'
        body=''
        if catalog==3:
            body=lowering.SOURCE.split('\n',1)[1].replace('fn main()->int','pub fn probe()->int').replace('ws://127.0.0.1:1/test',url)
            body+=f'\nshadow probe {{assert (== (probe) {expected})}}\n'
            calls.append(f'(part{index}.probe)')
        elif catalog==1:
            body='''pub fn probe()->int {
 match (temp) {Error(e)=>{assert false return 0} Ok(value)=>{
  match (close value) {Ok()=>{return 3} Error(e)=>{assert false return 0}}
 }}
}
shadow probe {assert (== (probe) 3)}
'''
            calls.append(f'(part{index}.probe)')
        path.write_text(declaration+body);dependencies.append(path)
        imports.append(f'module "{path.name}" as part{index}')
        flag={1:FLAGS[1],2:FLAGS[2],3:WS}[catalog]
        if flag not in flags:flags.append(flag)
    total=catalogs.count(3)*expected+catalogs.count(1)*3
    root=work/'source.nano'
    root.write_text('\n'.join(imports)+'\nfn main()->int {let mut value:int=0\n'+
        '\n'.join(f'set value (+ value {call})' for call in calls)+
        f'\nreturn value}}\nshadow main {{assert (== (main) {total})}}\n')
    return root,dependencies,flags,total


class MixedWebSocketProducts(unittest.TestCase):
    setUp=drivers.WebSocketServiceDrivers.setUp
    run_command=drivers.WebSocketServiceDrivers.run_command
    server=dispatch.WebSocketDispatch.server

    def test_permissions_shadows_and_native_invocation(self):
        for catalogs in [(3,1,2,3),(3,3,3,3,3)]:
            source,_,flags,expected=graph(self.work,catalogs)
            wires=[]
            for driver in self.drivers:
                wire=self.work/(driver.name+'.nvm');native=self.work/(driver.name+'.native')
                for denied in [[],[f for f in flags if f!=WS]]:
                    wire.write_bytes(b'prior')
                    self.run_command([driver,source,*denied,'--emit-nvm','-o',wire],1)
                    self.assertEqual(wire.read_bytes(),b'prior')
                self.run_command([driver,source,*flags,'--emit-nvm','-o',wire]);wires.append(wire.read_bytes())
                route=[ROOT/'bin/nano_vm','--services','--service-instruction-limit','100000']
                self.run_command([*route,*flags,wire],expected)
                self.run_command([*route,*[f for f in flags if f!=WS],wire],1)
                self.run_command([driver,source,*flags,'-o',native])
                self.run_command([native,*flags],expected)
                self.run_command([native,*[f for f in flags if f!=WS]],1)
                for extra in [[WS],['--allow-websocket-lookup'],['--websocket-resolver-helper','/unused'],
                              ['--allow-websocket-lookup','--websocket-resolver-helper','relative'],
                              ['--websocket-instruction-limit','100']]:
                    self.run_command([native,*flags,*extra],1)
                    self.run_command([*route,*flags,*extra,wire],1)
                if driver.name=='nano_virt':self.run_command([driver,source,*flags,'--run'],expected)
            for wire in wires[1:]:self.assertEqual(wires[0],wire)

    def test_nonexecuting_emission_and_failed_shadow_preserve_output(self):
        source,deps,flags,_=graph(self.work)
        original={path:path.read_text() for path in [*deps,source]}
        for driver in self.drivers:
            wire=self.work/(driver.name+'.nvm')
            for path,text in original.items():path.write_text('\n'.join(line for line in text.splitlines() if not line.startswith('shadow ')))
            self.run_command([driver,source,'--emit-nvm','-o',wire])
            self.run_command([driver,source,'-o',self.work/(driver.name+'.native')])
            for path,text in original.items():path.write_text(text)
            deps[0].write_text(original[deps[0]].replace('assert (== (probe) 2)','assert false'))
            wire.write_bytes(b'prior')
            self.run_command([driver,source,*flags,'--emit-nvm','-o',wire],1)
            self.assertEqual(wire.read_bytes(),b'prior')
            deps[0].write_text(original[deps[0]])

    def test_lookup_policy_is_supplied_at_invocation(self):
        source,deps,flags,expected=graph(self.work)
        for path in [*deps,source]:
            text='\n'.join(line for line in path.read_text().splitlines() if not line.startswith('shadow '))
            path.write_text(text.replace('Error(e)=>{return e.status}',
                'Error(e)=>{if (== e.status 2) {return 2} return 9}'))
        marker=self.work/'resolver-called'
        helper=self.work/'resolver-helper'
        helper.write_text('#!/bin/sh\nprintf x >> '+shlex.quote(str(marker))+'\nexit 1\n')
        helper.chmod(0o700)
        lookup=['--allow-websocket-lookup','--websocket-resolver-helper',str(helper)]
        for driver in self.drivers:
            native=self.work/(driver.name+'.lookup');wire=self.work/(driver.name+'.lookup.nvm')
            self.run_command([driver,source,*flags,*lookup,'-o',native])
            self.run_command([driver,source,*flags,*lookup,'--emit-nvm','-o',wire])
            self.assertFalse(marker.exists())
            route=[ROOT/'bin/nano_vm','--services','--service-instruction-limit','100000',wire]
            self.run_command([native,*flags],expected)
            self.run_command([*route,*flags],expected)
            self.assertFalse(marker.exists())
            self.run_command([native,*flags,*lookup],21)
            self.run_command([*route,*flags,*lookup],21)
            self.assertEqual(marker.read_bytes(),b'xxxx')
            marker.unlink()

    def test_relocated_native_product(self):
        source,_,flags,expected=graph(self.work)
        prefix=self.work/'installed'
        self.run_command(['make','-f','Makefile.gnu','-C',ROOT,'PREFIX='+str(prefix),'install-services-public-runtime'])
        (prefix/'bin').mkdir()
        for name in ['nanoc_c','nano_virt']:shutil.copy2(ROOT/'bin'/name,prefix/'bin'/name)
        moved=self.work/'relocated';prefix.rename(moved)
        env=dict(self.env);env.pop('NANO_ROOT',None)
        for name in ['nanoc_c','nano_virt']:
            native=self.work/(name+'.installed')
            self.run_command([moved/'bin'/name,source,*flags,'-o',native],env=env)
            self.run_command([native,*flags],expected)

    def test_required_live_peer(self):
        self.bad_reply=False
        port,messages=self.server()
        count=0
        for catalogs in [(3,1,2,3),(3,3,3,3,3)]:
            for host in ['127.0.0.1','localhost']:
                source,_,flags,expected=graph(self.work,catalogs,url=f'ws://{host}:{port}/test',expected=77)
                if host=='localhost':flags+=['--allow-websocket-lookup','--websocket-resolver-helper',str(ROOT/'bin/nano-resolver')]
                for driver in self.drivers:
                    wire=self.work/(driver.name+'.nvm');native=self.work/(driver.name+'.native')
                    self.run_command([driver,source,*flags,'--emit-nvm','-o',wire])
                    self.run_command([ROOT/'bin/nano_vm','--services','--service-instruction-limit','100000',*flags,wire],expected&255)
                    self.run_command([driver,source,*flags,'-o',native])
                    self.run_command([native,*flags],expected&255)
                    count+=12*catalogs.count(3)
        self.assertEqual(messages,[b'a\0b']*count)


class MixedWebSocketLowering(unittest.TestCase):
    command=staticmethod(base.ServiceLowering.command)
    lower=nano.ServiceLoweringNano.lower
    setUpClass=classmethod(lowering.WebSocketSourceLowering.setUpClass.__func__)

    def test_paired_source_bytes(self):
        for catalogs in [(3,1,2,3),(3,3,3,3,3)]:
            source,deps,_,expected=graph(self.work,catalogs)
            for indirect in [False,True]:
                if indirect:
                    for path in deps:
                        path.write_text(path.read_text().replace('let message:Message=(payload)',
                            'let factory:fn()->Message=payload let message:Message=(factory)'))
                native=self.work/'source.c'
                report=self.command([ROOT/'obj/test_service_lowering',source,'main',native])
                self.assertIn(f'EXEC 0 VALUE {expected}',report)
                self.assertEqual(self.lower(source,dependencies=deps),Path(str(native)+'.nvm').read_bytes())
