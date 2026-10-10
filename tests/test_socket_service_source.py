"""I compare TCP body and affine ownership facts before wire/runtime admission."""
import subprocess
import unittest
from tests import test_service_bodies as bodies
from tests import test_service_ownership as ownership

ROOT=bodies.ROOT
DECL=bodies.DECL.replace('filesystem','net')
ENDPOINT='Endpoint { family: 4, address0: 2130706433, address1: 0, address2: 0, address3: 0, port: 12345, scope_id: 0 }'
ADDRESS='pure fn address()->Endpoint { return '+ENDPOINT+' }\nshadow address { let value:Endpoint=(address) assert (== value.family 4) }\n'

def tcp(text):
    for old,new in [('FileError','SocketError'),('OpenResult','ConnectResult'),('WriteResult','SendResult'),('PositionResult','ConnectStatus'),('ReadResult','ReceiveResult'),('File','Conn'),('write_byte','send_byte'),('read_byte','receive_byte'),('rewind','finish_connect')]:
        text=text.replace(old,new)
    return text.replace('(temp)','(begin_connect (address))')

def source(text):
    return DECL+ADDRESS+text+('' if 'fn main(' in text else '\nfn main()->int{return 0}\n')

class SocketBodies(unittest.TestCase):
    run_command=staticmethod(bodies.ServiceBodies.run_command)
    check=bodies.ServiceBodies.check
    check_drivers=bodies.ServiceBodies.check_drivers

    @classmethod
    def setUpClass(cls):
        bodies.ServiceBodies.setUpClass.__func__(cls)
        (cls.work/'interface.nsi.json').write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())

    def test_endpoint_calls_results_and_shadows(self):
        self.check(source(tcp(bodies.POSITIVE)),0)
        self.check(source('fn port(value:Endpoint)->int{return value.port}'),0)
        self.check(source('pure fn copy(value:Endpoint)->Endpoint{return value}'),0)
        # Fields evaluate in source order, while names determine their identity.
        reordered='Endpoint { scope_id: 0, port: 12345, address3: 0, address2: 0, address1: 0, address0: 2130706433, family: 4 }'
        self.check(source('fn other()->Endpoint{return '+reordered+'}'),0)

    def test_exact_constructor_and_nominal_refusals(self):
        cases={
            'missing':ENDPOINT.replace(', scope_id: 0',''),
            'extra':ENDPOINT.replace('scope_id: 0','scope_id: 0, extra: 0'),
            'duplicate':ENDPOINT.replace('scope_id: 0','family: 0'),
            'unknown':ENDPOINT.replace('scope_id: 0','missing: 0'),
            'wrong-type':ENDPOINT.replace('family: 4','family: true'),
            'wrong-value-type':'0',
        }
        for name,value in cases.items():
            with self.subTest(case=name):self.check(source('fn bad()->Endpoint{return '+value+'}'),1)
        cases={
            'fabricated':'fn bad()->Conn{return Conn {}}',
            'fabricated-result':'fn bad()->ConnectResult{return ConnectResult {}}',
            'wrong-result':'fn bad()->ReceiveResult{return (begin_connect (address))}',
            'method-arity':'fn bad()->ConnectResult{return (begin_connect)}',
            'wrong-endpoint':'fn bad()->ConnectResult{return (begin_connect 0)}',
            'wrong-borrow':'fn bad(value:Conn)->SendResult{return (send_byte value 1)}',
            'immutable-borrow':'fn bad(value:Conn)->SendResult{return (send_byte &mut value 1)}',
            'pure-connect':'pure fn bad(value:Endpoint)->ConnectResult{return (begin_connect value)}',
            'missing-arm':'fn bad(value:ConnectResult)->void{match value{Ok(conn)=>{}}}',
            'missing-owner':'fn bad(value:ConnectResult)->void{match value{Ok()=>{} Error(e)=>{}}}',
            'bad-field':'fn bad(value:SocketError)->int{return value.missing}',
            'borrow-escape':'fn bad(value:&mut Conn)->Conn{return value}',
            'selected-shadow':'shadow main {let bad:Conn=0}',
        }
        for name,text in cases.items():
            with self.subTest(case=name):self.check(source(text),1)

class SocketOwnership(unittest.TestCase):
    checked=staticmethod(ownership.ServiceOwnership.checked)
    check_drivers=ownership.ServiceOwnership.check_drivers

    @classmethod
    def setUpClass(cls):
        ownership.ServiceOwnership.setUpClass.__func__(cls)
        (cls.work/'interface.nsi.json').write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())

    def check(self,text,status):
        return ownership.ServiceOwnership.check(self,source(text),status,complete=True)

    def test_moves_branches_loops_and_borrows(self):
        for name,text in ownership.VALID.items():
            with self.subTest(case=name):self.check(tcp(text),0)
        self.check('fn close_port(value:Conn)->int{let c:CloseResult=(close value) return 12345} fn build(value:Conn)->Endpoint{return '+ENDPOINT.replace('port: 12345','port: (close_port value)')+'}',0)
        self.check('fn return_result(value:ConnectResult)->ConnectResult{return value}',0)

    def test_callable_transport_and_borrowed_signatures(self):
        self.check('fn keep(value:Conn)->Conn{return value} '
                   'fn apply(f:fn(Conn)->Conn,value:Conn)->Conn{return (f value)} '
                   'fn exercise(value:Conn)->void{let f:fn(Conn)->Conn=keep let owned:Conn=(apply f value) let c:CloseResult=(close owned)}',0)
        self.check('fn carry(value:ConnectResult)->ConnectResult{return value} '
                   'fn apply(f:fn(ConnectResult)->ConnectResult,value:ConnectResult)->ConnectResult{return (f value)} '
                   'fn exercise(value:ConnectResult)->ConnectResult{let f:fn(ConnectResult)->ConnectResult=carry return (apply f value)}',0)
        self.check('fn send(value:&mut Conn,n:int)->SendResult{return (send_byte &mut value n)} '
                   'fn exercise(value:Conn)->void{let mut owned:Conn=value let f:fn(&mut Conn,int)->SendResult=send '
                   'let sent:SendResult=(f &mut owned 1) let c:CloseResult=(close owned)}',0)

    def test_invalid_ownership_and_constructor_effects(self):
        for name,text in ownership.INVALID.items():
            with self.subTest(case=name):self.check(tcp(text),1)
        prefix='fn close_port(value:Conn)->int{let c:CloseResult=(close value) return 12345} '
        doubled=ENDPOINT.replace('port: 12345','port: (close_port value)').replace('scope_id: 0','scope_id: (close_port value)')
        self.check(prefix+'fn build(value:Conn)->Endpoint{return '+doubled+'}',1)

    def test_lowering_does_not_reinterpret_tcp_as_file(self):
        path=self.work/'body.nano';path.write_text(source(tcp(bodies.POSITIVE)))
        for command in self.drivers:
            output=self.work/'prior.nvm';output.write_bytes(b'prior-output')
            run=subprocess.run(list(map(str,[*command,path,'-o',output,'--allow-temporary-files'])),cwd=ROOT,text=True,capture_output=True,timeout=90)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('I have not connected TCP wire and runtime lowering',run.stdout+run.stderr)
            self.assertEqual(output.read_bytes(),b'prior-output')

if __name__=='__main__':unittest.main()
