"""I exercise invocation ownership around real counted WebSocket traffic."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests import test_websocket_client as peers


class WebSocketValues(unittest.TestCase):
    peer_case = peers.WebSocketClient.peer_case

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nano-websocket-values-")
        cls.binary = Path(cls.tmp.name) / "values"
        command = shlex.split(os.environ.get("NANO_WEBSOCKET_CC", "cc"))
        command += ["-std=c11", "-D_GNU_SOURCE", "-Wall", "-Wextra", "-Werror", "-g"]
        command += shlex.split(os.environ.get("NANO_WEBSOCKET_CFLAGS", ""))
        command += [str(peers.ROOT / source) for source in (
            "tests/websocket_values_probe.c", "src/nsi_websocket_values.c",
            "src/nsi_websocket_transport.c", "src/nsi_websocket_protocol.c",
            "src/nsi_socket.c", "src/nsi_socket_resolver.c", "src/nsi_cap.c", "src/utf8.c")]
        command += shlex.split(subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "openssl"], text=True))
        subprocess.run(command + ["-o", str(cls.binary)], check=True, timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def traffic(self, peer):
        self.assertEqual(peers.client_frame(peer)[:2], (2, b"a\0b"))
        peer.sendall(b"\x82\x03a\0b")

    def test_move_borrow_counted_message_and_close(self):
        def behavior(peer):
            self.traffic(peer)
            self.assertEqual(peers.client_frame(peer)[:2], (8, b""))
            peer.sendall(b"\x88\0")
        self.peer_case("normal", behavior, hostname="localhost")

    def test_close_consumes_invalid_deadline(self):
        def behavior(peer):
            self.traffic(peer)
            self.assertEqual(peer.recv(1), b"")
        self.peer_case("invalid-close", behavior)

    def test_finish_reclaims_unhandled_connect_result(self):
        self.peer_case("unhandled", lambda peer: self.assertEqual(peer.recv(1), b""))

    def test_finish_reclaims_borrowed_connection(self):
        self.peer_case("borrowed", lambda peer: self.assertEqual(peer.recv(1), b""))
