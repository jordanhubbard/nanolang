"""I check counted transport ownership and failure behavior against real peers."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import unittest
from tests import test_websocket_client as peers

ROOT = peers.ROOT


class WebSocketTransport(unittest.TestCase):
    peer_case = peers.WebSocketClient.peer_case

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nano-websocket-transport-")
        cls.binaries = []
        crypto = shlex.split(subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "openssl"], text=True))
        for inject in (False, True):
            binary = Path(cls.tmp.name) / ("injected" if inject else "linked")
            command = shlex.split(os.environ.get("NANO_WEBSOCKET_CC", "cc"))
            command += ["-std=c11", "-D_GNU_SOURCE", "-Wall", "-Wextra", "-Werror", "-g"]
            command += shlex.split(os.environ.get("NANO_WEBSOCKET_CFLAGS", ""))
            sources = ["tests/websocket_transport_probe.c", "src/nsi_websocket_protocol.c",
                       "src/nsi_socket.c", "src/nsi_socket_resolver.c", "src/nsi_cap.c", "src/utf8.c"]
            if inject:
                command += ["-DNL_WS_INJECT"]
            else:
                sources += ["src/nsi_websocket_transport.c"]
            command += [str(ROOT / source) for source in sources]
            subprocess.run(command + crypto + ["-o", str(binary)], check=True, timeout=60)
            cls.binaries.append(binary)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_peers(self, mode, behavior):
        for i, binary in enumerate(self.binaries):
            with self.subTest(binary=binary.name):
                self.binary = binary
                self.peer_case(mode, behavior, hostname="localhost" if i else None)

    def test_counted_messages_survive_receive_and_close(self):
        def behavior(peer):
            for expected in ((1, b"a\0b"), (2, b"\0\xff\1"), (1, b"")):
                self.assertEqual(peers.client_frame(peer)[:2], expected)
            peer.sendall(b"\x81\x03a\0b\x82\x03\0\xff\1\x81\0\x88\x02\x03\xe8")
            self.assertEqual(peers.client_frame(peer)[:2], (8, b"\x03\xe8"))
        self.run_peers("counted", behavior)

    def test_partial_input_survives_timeout(self):
        def behavior(peer):
            peer.sendall(b"\x81\x05he")
            time.sleep(0.1)
            peer.sendall(b"llo")
            self.assertEqual(peers.client_frame(peer)[:2], (8, b""))
            peer.sendall(b"\x88\0")
        self.run_peers("partial", behavior)

    def test_protocol_failure_is_terminal(self):
        def behavior(peer):
            peer.sendall(b"\x83\0")
            self.assertEqual(peer.recv(1), b"")
        self.run_peers("invalid", behavior)

    def test_close_consumes_on_invalid_deadline(self):
        self.run_peers("close-invalid", lambda peer: self.assertEqual(peer.recv(1), b""))

    def test_close_consumes_on_timeout(self):
        def behavior(peer):
            self.assertEqual(peers.client_frame(peer)[:2], (8, b""))
            self.assertEqual(peer.recv(1), b"")
        self.run_peers("close-timeout", behavior)

    def test_partial_send_failure_terminates_stream(self):
        def behavior(peer):
            received = bytearray()
            while True:
                chunk = peer.recv(65536)
                if not chunk:
                    break
                received.extend(chunk)
            self.assertGreater(len(received), 0)
            self.assertLess(len(received), 1024 * 1024)
        self.run_peers("send-partial", behavior)
