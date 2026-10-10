"""I exercise my production WebSocket wrapper against real local RFC 6455 peers."""
import base64
import hashlib
import os
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def exact(peer, size):
    result = bytearray()
    while len(result) < size:
        chunk = peer.recv(size - len(result))
        if not chunk:
            raise AssertionError("I received premature EOF")
        result.extend(chunk)
    return bytes(result)


def client_frame(peer):
    first, second = exact(peer, 2)
    assert first & 0x80 and not first & 0x70 and second & 0x80
    size = second & 127
    if size == 126:
        size = struct.unpack("!H", exact(peer, 2))[0]
        assert size >= 126
    elif size == 127:
        size = struct.unpack("!Q", exact(peer, 8))[0]
        assert 65535 < size <= 1024 * 1024
    mask = exact(peer, 4)
    payload = exact(peer, size)
    return first & 15, bytes(b ^ mask[i % 4] for i, b in enumerate(payload)), mask


class WebSocketClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nano-websocket-client-")
        cls.binary = Path(cls.tmp.name) / "client"
        compiler = shlex.split(os.environ.get("NANO_WEBSOCKET_CC", "cc"))
        flags = shlex.split(os.environ.get("NANO_WEBSOCKET_CFLAGS", ""))
        crypto = subprocess.check_output(["pkg-config", "--cflags", "--libs", "openssl"], text=True)
        command = compiler + ["-std=c11", "-D_GNU_SOURCE", "-Wall", "-Wextra", "-Werror", "-g"] + flags
        command += [str(ROOT / p) for p in (
            "tests/websocket_client_probe.c", "modules/websocket/websocket_helpers.c",
            "src/nsi_websocket_protocol.c", "src/nsi_socket.c", "src/nsi_socket_resolver.c", "src/nsi_cap.c", "src/utf8.c")]
        command += shlex.split(crypto) + ["-o", str(cls.binary)]
        subprocess.run(command, check=True, timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def peer_case(self, mode, behavior, bad_upgrade=None, family=socket.AF_INET, hostname=None):
        listener = socket.socket(family)
        listener.settimeout(5)
        listener.bind(("::1" if family == socket.AF_INET6 else "127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        host = hostname or ("[::1]" if family == socket.AF_INET6 else "127.0.0.1")
        errors = []

        def serve():
            try:
                with listener.accept()[0] as peer:
                    peer.settimeout(5)
                    request = bytearray()
                    while not request.endswith(b"\r\n\r\n"):
                        request.extend(exact(peer, 1))
                        self.assertLess(len(request), 8192)
                    lines = bytes(request).decode("ascii").split("\r\n")
                    self.assertEqual(lines[0], "GET /?mode=test HTTP/1.1")
                    headers = dict(line.split(": ", 1) for line in lines[1:] if line)
                    self.assertEqual(headers["Host"], f"{host}:{port}")
                    key = headers["Sec-WebSocket-Key"]
                    self.assertEqual(len(base64.b64decode(key, validate=True)), 16)
                    accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
                    if bad_upgrade is not None:
                        peer.sendall(bad_upgrade)
                    else:
                        peer.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n")
                        behavior(peer)
            except BaseException as error:
                errors.append(error)
            finally:
                listener.close()

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            env = dict(os.environ, NANOLANG_RESOLVER=str(ROOT / "bin/nano-resolver"))
            result = subprocess.run([str(self.binary), f"ws://{host}:{port}?mode=test", mode], env=env, capture_output=True, text=True, timeout=15)
        finally:
            worker.join(timeout=6)
            listener.close()
        self.assertFalse(worker.is_alive(), "I left my local peer running")
        if errors:
            raise errors[0]
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_long_frames_random_masks_fragments_and_control_payload(self):
        def behavior(peer):
            first = client_frame(peer)
            second = client_frame(peer)
            self.assertEqual(first[:2], (1, b"x" * 70000))
            self.assertEqual(second[:2], first[:2])
            self.assertNotEqual(first[2], second[2])
            # I split one UTF-8 scalar across continuation frames and interleave a ping.
            peer.sendall(b"\x01\x06hello\xe2\x89\x03p\x00g\x80\x02\x82\xac")
            self.assertEqual(client_frame(peer)[:2], (10, b"p\x00g"))
            peer.sendall(b"\x88\x02\x03\xe8")
            self.assertEqual(client_frame(peer)[:2], (8, b"\x03\xe8"))
        self.peer_case("normal", behavior)

    def test_partial_frame_survives_timeout(self):
        def behavior(peer):
            peer.sendall(b"\x81\x05he")
            time.sleep(0.15)
            peer.sendall(b"llo")
            self.assertEqual(client_frame(peer)[:2], (8, b""))
            peer.sendall(b"\x88\x00")
        self.peer_case("partial", behavior)

    def test_close_has_valid_length_and_mask(self):
        def behavior(peer):
            self.assertEqual(client_frame(peer)[:2], (8, b""))
            peer.sendall(b"\x88\x00")
            self.assertEqual(peer.recv(1), b"")  # I do not send a second close.
        self.peer_case("close", behavior)

    def test_ipv6_url_and_hostname_resolution(self):
        def close_frame(peer):
            self.assertEqual(client_frame(peer)[:2], (8, b""))
            peer.sendall(b"\x88\x00")
        self.peer_case("close", close_frame, family=socket.AF_INET6)
        self.peer_case("close", close_frame, hostname="localhost")

    def test_invalid_server_frames_close_transport(self):
        for frame in (b"\xc1\x00", b"\x81\x80", b"\x81\x02\xc0\xaf", b"\x88\x01x", b"\x80\x00"):
            with self.subTest(frame=frame):
                def behavior(peer):
                    peer.sendall(frame)
                    self.assertEqual(peer.recv(1), b"")
                self.peer_case("invalid", behavior)

    def test_dns_helper_failure_and_deadline(self):
        env = dict(os.environ, NANOLANG_RESOLVER="/nonexistent-nanolang-resolver")
        command = [str(self.binary), "ws://localhost:9/", "refuse"]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        helper = Path(self.tmp.name) / "hung resolver"
        helper.write_text("#!/bin/sh\nexec sleep 30\n")
        helper.chmod(0o755)
        env["NANOLANG_RESOLVER"] = str(helper)
        start = time.monotonic()
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=14)
        elapsed = time.monotonic() - start
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(elapsed, 9)
        self.assertLess(elapsed, 13)

    def test_forged_or_unverified_upgrade_is_refused(self):
        for response in (
            b"HTTP/1.1 403 Forbidden\r\nX-Trace: 101\r\n\r\n",
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: AAAAAAAAAAAAAAAAAAAAAAAAAAA=\r\n\r\n",
        ):
            with self.subTest(response=response):
                self.peer_case("refuse", None, response)


if __name__ == "__main__":
    unittest.main()
