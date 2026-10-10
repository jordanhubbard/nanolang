"""I exercise the private checked carrier against real WebSocket peers."""
import os
from pathlib import Path
from tests import test_websocket_values as values


class WebSocketRuntime(values.WebSocketValues):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ["NANO_WEBSOCKET_RUNTIME"])

    @classmethod
    def tearDownClass(cls):
        pass

    def test_native_storage_and_permuted_imports(self):
        def behavior(peer):
            self.traffic(peer)
            self.assertEqual(values.peers.client_frame(peer)[:2], (8, b""))
            peer.sendall(b"\x88\0")
        self.peer_case("native", behavior)

    def test_receive_allocation_failure_drains_connection(self):
        self.receive_failure("allocation-receive")

    def test_receive_storage_limit_drains_connection(self):
        self.receive_failure("budget-receive")

    def receive_failure(self, mode):
        if "instrumented" not in self.binary.name:
            self.skipTest("I inject private runtime allocation and accounting failures.")
        def behavior(peer):
            self.assertEqual(values.peers.client_frame(peer)[:2], (2, b"a\0b"))
            peer.sendall(b"\x82\x03c\0d")
            self.assertEqual(peer.recv(1), b"")
        self.peer_case(mode, behavior)
