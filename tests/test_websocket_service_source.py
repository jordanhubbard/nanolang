"""I compare independent WebSocket source type/body facts before wire lowering."""
import unittest
from tests import test_service_bodies as bodies

ROOT = bodies.ROOT
DECL = bodies.DECL.replace('filesystem', 'websocket')
MESSAGE = 'Message { binary: true, data: "a\\0b" }'

def source(text):
    return DECL + text + ('' if 'fn main(' in text else '\nfn main()->int{return 0}\n')

class WebSocketBodies(unittest.TestCase):
    run_command = staticmethod(bodies.ServiceBodies.run_command)
    check = bodies.ServiceBodies.check
    check_drivers = bodies.ServiceBodies.check_drivers

    @classmethod
    def setUpClass(cls):
        bodies.ServiceBodies.setUpClass.__func__(cls)
        (cls.work / 'interface.nsi.json').write_bytes((ROOT / 'tests/fixtures/nsi_websocket_plan.json').read_bytes())

    def test_message_strings_and_complete_lifecycle(self):
        self.check(source('fn payload()->Message{return ' + MESSAGE + '}\n'
            'fn text(value:Message)->string{return value.data}\n'
            'fn emit(value:&mut Connection,message:Message)->SendResult{return (send &mut value message 1000)}\n'
            'fn main()->int { let opened:ConnectResult=(connect "ws://localhost/test" 1000) '
            'match opened { Ok(conn)=>{let mut owned:Connection=conn '
            'let sent:SendResult=(emit &mut owned (payload)) '
            'match sent {Ok(n)=>{assert (== n 3)} Error(e)=>{assert (>= e.status 0)}} '
            'let received:ReceiveResult=(receive &mut owned 1000) '
            'match received {Ok(message)=>{let data:string=message.data assert message.binary} Error(e)=>{assert (not e.terminal)}} '
            'let closed:CloseResult=(close owned 1000) '
            'match closed {Ok()=>{} Error(e)=>{assert (>= e.close_code 0)}}} '
            'Error(e)=>{assert (>= e.resolver_error 0)}} return 0}\nshadow main{assert true}'), 0)
        self.check(source('pure fn reordered()->Message{return Message {data:"", binary:false}}'), 0)

    def test_invalid_shapes_signatures_and_owners(self):
        cases = [
            'fn bad()->Message{return Message {binary:true}}',
            'fn bad()->Message{return Message {binary:true,data:1}}',
            'fn bad()->Message{return Message {binary:0,data:"x"}}',
            'fn bad()->Message{return Message {binary:true,binary:false}}',
            'fn bad()->Message{return Message {binary:true,data:"x",extra:1}}',
            'fn bad()->Connection{return Connection {}}',
            'fn bad()->ConnectResult{return (connect 123 1000)}',
            'fn bad()->ConnectResult{return (connect "ws://localhost" true)}',
            'fn bad(value:&mut Connection,message:Message)->SendResult{return (send &mut value message)}',
            'fn bad(value:&mut Connection)->SendResult{return (send &mut value "x" 1000)}',
            'fn bad(value:&mut Connection)->ReceiveResult{return (receive &mut value false)}',
            'fn bad(value:Connection)->CloseResult{return (close value)}',
            'fn bad(value:Message)->int{return value.data}',
            'pure fn bad()->ConnectResult{return (connect "ws://localhost" 1000)}',
        ]
        for text in cases:
            with self.subTest(source=text):
                self.check(source(text), 1)

from tests import test_service_ownership as ownership

class WebSocketOwnership(unittest.TestCase):
    checked = staticmethod(ownership.ServiceOwnership.checked)
    check_drivers = ownership.ServiceOwnership.check_drivers

    @classmethod
    def setUpClass(cls):
        ownership.ServiceOwnership.setUpClass.__func__(cls)
        (cls.work / 'interface.nsi.json').write_bytes((ROOT / 'tests/fixtures/nsi_websocket_plan.json').read_bytes())

    def check(self, text, status):
        return ownership.ServiceOwnership.check(self, source(text), status, complete=True)

    def test_loops_branches_results_and_borrowed_helpers(self):
        self.check('fn emit(value:&mut Connection,message:Message)->SendResult{return (send &mut value message 1000)} '
            'fn exercise(value:Connection)->void {let mut owned:Connection=value let mut i:int=0 '
            'while (< i 2) {let sent:SendResult=(emit &mut owned '+MESSAGE+') set i (+ i 1)} '
            'let closed:CloseResult=(close owned -1)}', 0)
        self.check('fn exercise(value:Connection,yes:bool)->void {if yes {let closed:CloseResult=(close value 1000)} '
            'else {let closed:CloseResult=(close value 0)}}', 0)
        self.check('fn exercise(value:ConnectResult)->void {match value {Error(e)=>{} Ok(conn)=>{let closed:CloseResult=(close conn 1000)}}}', 0)

    def test_consumed_and_unhandled_owners(self):
        for text in [
            'fn exercise(value:Connection)->void {let first:CloseResult=(close value 1000) let second:CloseResult=(close value 1000)}',
            'fn exercise(value:Connection)->void {return}',
            'fn exercise()->void {let opened:ConnectResult=(connect "ws://localhost" 1000)}',
            'fn exercise(value:Connection,yes:bool)->void {if yes {let closed:CloseResult=(close value 1000)}}',
        ]:
            with self.subTest(source=text):
                self.check(text, 1)
