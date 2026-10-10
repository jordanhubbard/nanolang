"""I retain the exact WebSocket artifact ABI and borrowed-string snapshots."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests.test_websocket_client import WebSocketClient, client_frame

ROOT = Path(__file__).resolve().parents[1]


class WebSocketArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nano-websocket-artifacts-"))
        print("I retain WebSocket artifact evidence at", cls.work, flush=True)
        cls.env = dict(os.environ, NANO_BUILD_CACHE=str(cls.work / "cache"))
        cls.sequence = 0

    def command(self, command, expected=0):
        command = list(map(str, command))
        type(self).sequence += 1
        result = subprocess.run(command, cwd=ROOT, env=self.env, capture_output=True, text=True, timeout=180)
        (self.work / f"command-{self.sequence}.log").write_text(repr(command)+"\n"+result.stdout+result.stderr+f"\nexit={result.returncode}\n")
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def check_driver(self, compiler, label):
        module = self.work / (label + ".nvm")
        self.command(compiler + [ROOT / "tests/fixtures/websocket_artifact_client.nano", "--emit-nvm", "-o", module])
        source, native = module.with_suffix(".c"), module.with_suffix(".native")
        self.command([ROOT / "bin/nvm2c", module, "-o", source])
        cc = shlex.split(os.environ.get("NANO_NATIVE_TEST_CC", "cc"))
        self.command(cc + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", "-fno-sanitize-recover=all", source, ROOT / "bin/nano_aot_runtime.o", "-ldl", "-lm", "-o", native] + shlex.split(os.environ.get("LDFLAGS", "")))
        def behavior(peer):
            self.assertEqual(client_frame(peer)[:2], (1, b"hello"))
            peer.sendall(b"\x81\x05first\x81\x06second")
            self.assertEqual(client_frame(peer)[:2], (8, b""))
            peer.sendall(b"\x88\x00")
        for name, command in (("vm", [ROOT / "bin/nano_vm", module, "--"]), ("native", [native])):
            with self.subTest(driver=label, route=name):
                self.binary = self.work / (label + "-" + name)
                self.binary.write_text("#!/bin/sh\nexec " + shlex.join(list(map(str, command))) + ' "$@"\n')
                self.binary.chmod(0o700)
                WebSocketClient.peer_case(self, "artifact", behavior)

    def test_seed_vm_and_native(self):
        self.check_driver([ROOT / "bin/nano_virt"], "seed")

    def test_updated_nano_vm_and_native(self):
        command = os.environ.get("NANO_WEBSOCKET_DRIVER")
        if not command:
            self.skipTest("I require an explicitly selected updated Nano compiler")
        self.check_driver(shlex.split(command), "nano")

    def test_native_refuses_wrong_abi_without_replacing_output(self):
        for signature in ('"nl_ws_connect" int int', '"nl_ws_receive" int int',
                          '"nl_ws_send" int int bool', '"nl_ws_close" int',
                          '"nl_ws_unknown" int int'):
            with self.subTest(signature=signature):
                assembly = self.work / "refusal.nasm"
                module, output = assembly.with_suffix(".nvm"), assembly.with_suffix(".c")
                assembly.write_text('.import "/missing/websocket.so" ' + signature +
                    '\n.import_kind 0 artifact\n.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
                self.command([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                output.write_bytes(b"prior output\n")
                result = self.command([ROOT / "bin/nvm2c", module, "-o", output], expected=1)
                self.assertIn("requires an exact library binding and typed value adapter", result.stderr)
                self.assertEqual(output.read_bytes(), b"prior output\n")

    def test_updated_nano_refuses_wrong_abi_without_replacing_output(self):
        command = os.environ.get("NANO_WEBSOCKET_DRIVER")
        if not command:
            self.skipTest("I require an explicitly selected updated Nano compiler")
        for declaration, call, diagnostic in (
            ('extern fn nl_ws_connect(a:int)->int', '(nl_ws_connect 0)', 'exact operands for this artifact call'),
            ('extern fn nl_ws_receive(a:int)->int', '(nl_ws_receive 0)', 'exact artifact result and arity'),
            ('extern fn nl_ws_close()->int', '(nl_ws_close)', 'exact artifact result and arity'),
        ):
            with self.subTest(declaration=declaration):
                source, output = self.work / "wrong.nano", self.work / "wrong.nvm"
                source.write_text(declaration + '\nfn main()->int { unsafe { let result:int = ' +
                    call + ' } return 0 }\nshadow main { assert true }\n')
                output.write_bytes(b"prior output\n")
                result = self.command(shlex.split(command) + [source, "--emit-nvm", "-o", output], expected=1)
                self.assertIn(diagnostic, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b"prior output\n")


if __name__ == "__main__":
    unittest.main()
