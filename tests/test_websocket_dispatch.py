"""I compare real WebSocket bytecode and separately compiled native execution."""
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import threading
import unittest
from tests import test_websocket_client as peers

ROOT = peers.ROOT


class WebSocketDispatch(unittest.TestCase):
    fixture_source = "tests/nanoisa/test_websocket_dispatch.c"
    dispatch_sources = ["src/nanovm/websocket_vm_indirect_private.c", "src/nanoisa/nvm2c_websocket_indirect_private.c"]
    extra_sources = []
    include_flags = []
    policy_expression = "runtime_policy(c,policy)"

    def driver_source(self, private_source):
        return private_source

    def command(self, name, args):
        (self.artifacts / (name + "-command.txt")).write_text(shlex.join(args) + "\n")
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=180,
            env=dict(os.environ, NANOLANG_RESOLVER=str(ROOT / "bin/nano-resolver"),
                     ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"))
        (self.artifacts / (name + ".log")).write_text(result.stdout + result.stderr)
        (self.artifacts / (name + "-status.txt")).write_text(str(result.returncode) + "\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def server(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(32)
        listener.settimeout(0.2)
        stop = threading.Event()
        errors, messages = [], []
        def serve():
            while not stop.is_set():
                try:
                    peer = listener.accept()[0]
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    with peer:
                        peer.settimeout(3)
                        request = bytearray()
                        while not request.endswith(b"\r\n\r\n"):
                            request.extend(peers.exact(peer, 1))
                            self.assertLess(len(request), 8192)
                        lines = bytes(request).decode("ascii").split("\r\n")
                        self.assertEqual(lines[0], "GET /?mode=test HTTP/1.1")
                        headers = dict(line.split(": ", 1) for line in lines[1:] if line)
                        accept = base64.b64encode(hashlib.sha1((headers["Sec-WebSocket-Key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
                        peer.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n")
                        while True:
                            if not peer.recv(1, socket.MSG_PEEK):
                                break
                            opcode, data, _ = peers.client_frame(peer)
                            if opcode == 8:
                                self.assertEqual(data, b"")
                                peer.sendall(b"\x88\0")
                                break
                            self.assertEqual((opcode, data), (2, b"a\0b"))
                            messages.append(data)
                            if self.bad_reply:
                                peer.sendall(b"\x81\x02\xc0\xaf")
                                break
                            peer.sendall(b"\x82\x03a\0b")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # I allow an invocation to terminate at an explicit fuel boundary.
                except BaseException as error:
                    errors.append(error)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        def cleanup():
            stop.set();listener.close();thread.join(4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
        self.addCleanup(cleanup)
        return listener.getsockname()[1], messages

    def build_fixture(self):
        self.artifacts = Path(tempfile.mkdtemp(prefix="nano-websocket-dispatch-"))
        print("I retain WebSocket dispatch artifacts at", self.artifacts, flush=True)
        compiler = shlex.split(os.environ.get("NANO_WEBSOCKET_CC", "cc"))
        flags = shlex.split(os.environ.get("NANO_WEBSOCKET_CFLAGS", ""))
        flags += ["-std=c11", "-D_GNU_SOURCE", "-Wall", "-Wextra", "-Werror", "-g", "-O1", "-I.",
                  "-DNVM_WEBSOCKET_INDIRECT_VM_PRIVATE", "-DNVM_WEBSOCKET_INDIRECT_NATIVE_PRIVATE"]
        if os.environ.get("NANO_WEBSOCKET_SANITIZERS", "1") != "0":
            flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        flags += self.include_flags
        flags += shlex.split(subprocess.check_output(["pkg-config", "--cflags", "openssl"], text=True))
        ordinary = shlex.split(os.environ["WEBSOCKET_DISPATCH_OBJECTS"])
        ldflags = shlex.split(os.environ.get("WEBSOCKET_DISPATCH_LDFLAGS", "-lm"))
        ldflags += shlex.split(subprocess.check_output(["pkg-config", "--libs", "openssl"], text=True))
        providers = []
        for source in ["src/nanoisa/websocket_flow.c", "src/nanoisa/websocket_codec.c", "src/nanoisa/websocket_runtime.c",
                       "src/nanoisa/service_websocket_nominal.c", "src/nanoisa/service_websocket_nominal_plan.c",
                       "src/nsi_websocket_plan.c", "src/nsi_websocket_values.c", "src/nsi_websocket_transport.c",
                       "src/nsi_websocket_protocol.c", "src/nsi_socket.c", "src/nsi_socket_resolver.c", "src/nsi_cap.c"] + self.extra_sources:
            obj = self.artifacts / (Path(source).stem + ".o")
            hooks = ["-include", "tests/nanoisa/websocket_dispatch_hooks.h", "-Dsocket=websocket_dispatch_socket", "-Dclose=websocket_dispatch_close"] if Path(source).stem == "nsi_socket" else []
            self.command(obj.stem + "-build", [*compiler, *flags, *hooks, "-c", source, "-o", str(obj)])
            providers.append(str(obj))
        fixture = self.artifacts / "fixture"
        self.command("fixture-build", [*compiler, *flags, self.fixture_source, *self.dispatch_sources,
            *providers, *ordinary, *ldflags, "-o", str(fixture)])
        return compiler, flags, ordinary, ldflags, providers, fixture

    def test_vm_and_generated_native(self):
        compiler, flags, ordinary, ldflags, providers, fixture = self.build_fixture()
        self.bad_reply = False
        port, messages = self.server()
        driver = self.artifacts / "driver.c"
        driver.write_text(self.driver_source('''#include "src/nanoisa/nvm2c_websocket_indirect_private.h"
#include <stdlib.h>
#include "tests/nanoisa/websocket_dispatch_host.h"
int main(int argc,char **argv){if(argc!=4)return 2;
 NvmWebSocketIndirectOptions options={1,strtoull(argv[1],NULL,10)};
 NlWsTransportPolicy policy={atoi(argv[2])!=0,atoi(argv[2])!=2,getenv("NANOLANG_RESOLVER"),2000};
 fail_close=atoi(argv[3])!=0;NvmWebSocketRuntimeView out={.fields=99,.values={12345}};
 NvmWebSocketIndirectExecutionReport r=nvm_websocket_native_indirect_execute(&options,&out,atoi(argv[2])<0?NULL:&policy);report(r,out);return 0;}
'''))
        for case in range(3):
            self.command("emit-" + str(case), [str(fixture), "emit", str(self.artifacts), str(port), str(case)])
            wire = self.artifacts / f"case-{case}.nvm"
            source = self.artifacts / f"case-{case}.c"
            text = source.read_text()
            self.assertIn("nf_function_", text)
            self.assertIn(self.policy_expression, text)
            self.assertNotIn("fvm_step", text)
            vm_cmd = [str(fixture), "vm", str(wire)]
            for optimization in ("-O0", "-O2"):
                native = self.artifacts / f"native-{case}-{optimization[1:]}"
                self.command(native.name + "-build", [*compiler, *flags, optimization, str(source), str(driver), *providers, *ordinary, *ldflags, "-o", str(native)])
                symbols = self.command(native.name + "-symbols", ["nm", str(native)])
                self.assertNotRegex(symbols, r"\b_?(nvm_websocket_vm_indirect_execute|nvm2c_websocket_indirect_private_emit|vm_execute|vm_core_execute)\b")
                for label, fuel, network, fault in [("normal", 100000, 1, 0), ("deny", 100000, 0, 0),
                    ("zero-fuel", 0, 1, 0), ("borrow-fuel", 20, 1, 0), ("receive-fuel", 40, 1, 0), ("close-failure", 100000, 1, 1),
                    ("missing-policy", 100000, -1, 0), ("protocol", 100000, 1, 0)]:
                    self.bad_reply = label == "protocol"
                    args = list(map(str, (fuel, network, fault)))
                    vm = json.loads(self.command(native.name + "-vm-" + label, vm_cmd + args))
                    actual = json.loads(self.command(native.name + "-" + label, [str(native), *args]))
                    self.assertEqual(actual, vm)
                    self.assertEqual(actual["opens"], actual["closes"])
                    if label == "missing-policy":
                        self.assertEqual((actual["status"], actual["acquired"], actual["opens"], actual["fields"], actual["steps"]), (1, 0, 0, 99, 0))
                    elif label == "protocol":
                        self.assertEqual((actual["status"], actual["fields"], actual["cleanup"]), (9, 99, 0))
                    elif "fuel" in label:
                        self.assertEqual((actual["status"], actual["fields"], actual["fuel"], actual["steps"]), (3, 99, 1, fuel))
                    elif fault:
                        self.assertEqual((actual["status"], actual["fields"]), (10, 99))
                        self.assertGreater(actual["cleanup"], 0)
                    else:
                        self.assertEqual(actual["status"], 0)
                        self.assertEqual(actual["value"], 2 if not network else 4 if case == 2 else 77)
                        self.assertEqual(actual["fields"], 1)
                self.bad_reply = False
            if case == 1:
                self.tamper(compiler, flags, providers, ordinary, ldflags, text, driver)
        malformed = self.artifacts / "truncated.nvm"
        malformed.write_bytes((self.artifacts / "case-0.nvm").read_bytes()[:-1])
        refused = json.loads(self.command("vm-truncated", [str(fixture), "vm", str(malformed), "100000", "1", "0"]))
        self.assertNotEqual(refused["status"], 0)
        self.assertEqual((refused["acquired"], refused["opens"], refused["fields"]), (0, 0, 99))
        self.command("emit-truncated", [str(fixture), "refuse", str(malformed)])
        self.assertGreaterEqual(len(messages), 24)

    def tamper(self, compiler, flags, providers, ordinary, ldflags, generated, driver):
        service = re.search(r"NF_TRY\(nvm_websocket_runtime_service\(c,(\d+),reference,inputs,2,dst\)\);", generated)
        self.assertIsNotNone(service)
        changed = service.group(0).replace("c," + service.group(1) + ",", "c," + str((int(service.group(1)) + 1) % 4) + ",")
        swapped = "{uint32_t temp=inputs[0];inputs[0]=inputs[1];inputs[1]=temp;}\n" + service.group(0)
        pending = next((m for m in re.finditer(r"if\(\(uint64_t\)\(variant.body.pending_checks\)!=UINT64_C\((\d+)\)\)return false;", generated) if int(m.group(1)) & 1024), None)
        self.assertIsNotNone(pending)
        edits = {
            "wrong-import": (service.group(0), changed, 6),
            "argument-order": (service.group(0), swapped, 6),
            "pending-mask": (pending.group(0), pending.group(0).replace("UINT64_C(" + pending.group(1) + ")", "UINT64_C(" + str(int(pending.group(1)) & ~1024) + ")"), 2),
        }
        for label, (old, new, status) in edits.items():
            self.assertNotEqual(old, new)
            source = self.artifacts / (label + ".c")
            source.write_text(generated.replace(old, new, 1))
            binary = self.artifacts / label
            self.command(label + "-build", [*compiler, *flags, str(source), str(driver), *providers, *ordinary, *ldflags, "-o", str(binary)])
            observed = json.loads(self.command(label, [str(binary), "100000", "1", "0"]))
            self.assertEqual((observed["status"], observed["opens"], observed["closes"], observed["fields"]), (status, 0, 0, 99))
