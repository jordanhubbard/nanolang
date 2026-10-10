"""I execute checked TCP bytecode and separately compiled native C over real loopback."""
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]

class SocketDispatch(unittest.TestCase):
    def command(self, name, args):
        (self.artifacts / f"{name}-command.txt").write_text(shlex.join(args) + "\n")
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=180,
                                env=dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                                         UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"))
        (self.artifacts / f"{name}.log").write_text(result.stdout + result.stderr)
        (self.artifacts / f"{name}-status.txt").write_text(str(result.returncode) + "\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def server(self, family):
        server = socket.socket(family, socket.SOCK_STREAM)
        server.bind(("127.0.0.1" if family == socket.AF_INET else "::1", 0))
        server.listen(32)
        server.settimeout(0.2)
        stop = threading.Event()
        received = []
        def serve():
            while not stop.is_set():
                try:
                    peer, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with peer:
                    peer.settimeout(3)
                    try:
                        data = peer.recv(1)
                        if data:
                            received.append(data)
                            peer.sendall(b"Z")
                    except (OSError, socket.timeout):
                        pass
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        def cleanup():
            stop.set()
            server.close()
            thread.join(4)
            self.assertFalse(thread.is_alive())
        self.addCleanup(cleanup)
        return server.getsockname()[1], received

    def test_vm_and_generated_native_tcp(self):
        self.artifacts = Path(tempfile.mkdtemp(prefix="nano-socket-dispatch-"))
        print(f"I retain TCP execution artifacts at {self.artifacts}", flush=True)
        compiler = shlex.split(os.environ.get("NANO_SOCKET_DISPATCH_CC", "cc"))
        flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-Wall", "-Wextra", "-Werror", "-g", "-I.",
                 "-DNVM_SOCKET_INDIRECT_VM_PRIVATE", "-DNVM_SOCKET_INDIRECT_NATIVE_PRIVATE"]
        if os.environ.get("NANO_SOCKET_DISPATCH_SANITIZERS", "1") != "0":
            flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        ordinary = shlex.split(os.environ["SOCKET_DISPATCH_OBJECTS"])
        ldflags = shlex.split(os.environ.get("SOCKET_DISPATCH_LDFLAGS", "-lm -lcrypto"))
        sources = ["src/nanoisa/service_socket_nominal.c", "src/nanoisa/service_socket_nominal_plan.c",
                   "src/nsi_file_plan.c", "src/nsi_socket_plan.c", "src/nanoisa/socket_flow.c",
                   "src/nanoisa/socket_runtime.c", "src/nsi_cap.c", "src/nsi_socket_values.c", "src/nsi_socket.c"]
        providers = []
        for source in sources:
            out = self.artifacts / (Path(source).stem + ".o")
            hooks = (["-include", "tests/nanoisa/socket_dispatch_hooks.h", "-Dsocket=dispatch_socket",
                      "-Dclose=dispatch_close"] if Path(source).stem == "nsi_socket" else [])
            self.command(out.stem + "-build", [*compiler, *flags, "-O1", *hooks, "-c", source, "-o", str(out)])
            providers.append(str(out))
        fixture = self.artifacts / "fixture"
        self.command("fixture-build", [*compiler, *flags, "-O1", "tests/nanoisa/test_socket_dispatch.c",
                     "src/nanovm/socket_vm_indirect_private.c", "src/nanoisa/nvm2c_socket_indirect_private.c",
                     *providers, *ordinary, *ldflags, "-o", str(fixture)])
        port4, received4 = self.server(socket.AF_INET)
        port6, received6 = self.server(socket.AF_INET6)
        self.command("emit", [str(fixture), "emit", str(self.artifacts), str(port4), str(port6)])
        expected = {0: 77, 1: 77, 2: 4, 3: None}
        for case in range(4):
            wire = self.artifacts / f"case-{case}.nvm"
            generated = (self.artifacts / f"case-{case}.c").read_text()
            self.assertIn("nf_function_", generated)
            self.assertIn("nvm_socket_runtime_indirect_enter(c)", generated)
            self.assertNotIn("fvm_step", generated)
            self.assertNotIn("nvm_socket_vm_indirect_execute", generated)
            reference = json.loads(self.command(f"vm-{case}", [str(fixture), "vm", str(wire), "100000", "0"]))
            self.validate(reference, expected[case], case)
            driver = self.artifacts / f"driver-{case}.c"
            driver.write_text('''#include "src/nanoisa/nvm2c_socket_indirect_private.h"
#include "tests/nanoisa/socket_dispatch_host.h"
int main(int argc,char **argv){
 if(argc!=3)return 2;
 NvmSocketIndirectOptions options={1,strtoull(argv[1],NULL,10)};
 dispatch_close_fault=atoi(argv[2])!=0;
 NvmSocketRuntimeView out={.fields=99,.values={12345}};
 NvmSocketIndirectExecutionReport r=nvm_socket_native_indirect_execute(&options,&out);
 dispatch_report(r,out);return 0;}
''')
            for opt in ("-O0", "-O2"):
                native = self.artifacts / f"native-{case}-{opt[1:]}"
                self.command(native.name + "-build", [*compiler, *flags, opt, str(self.artifacts / f"case-{case}.c"),
                             str(driver), *providers, *ordinary, *ldflags, "-o", str(native)])
                symbols = self.command(native.name + "-symbols", ["nm", str(native)])
                self.assertNotRegex(symbols, r"\b_?(nvm_socket_vm_indirect_execute|nvm2c_socket_indirect_private_emit|vm_execute|vm_core_execute)\b")
                actual = json.loads(self.command(native.name + "-run", [str(native), "100000", "0"]))
                self.validate(actual, expected[case], case)
                # Connection readiness can change iteration counts across executions.
                for key in ("status", "cleanup", "value", "fields", "opens", "closes"):
                    self.assertEqual(actual[key], reference[key])
                for fuel in (0, 25):
                    limited = json.loads(self.command(native.name + f"-fuel-{fuel}", [str(native), str(fuel), "0"]))
                    vm = json.loads(self.command(native.name + f"-vm-fuel-{fuel}", [str(fixture), "vm", str(wire), str(fuel), "0"]))
                    for key in ("status", "fuel", "steps", "fields", "value", "opens", "closes"):
                        self.assertEqual(limited[key], vm[key])
                    self.assertEqual(limited["opens"], limited["closes"])
                    if limited["fuel"]:
                        self.assertEqual(limited["steps"], fuel)
                        self.assertEqual(limited["fields"], 99)
                if case in (0, 1):
                    for label, command in (("native", [str(native), "100000", "1"]),
                                           ("vm", [str(fixture), "vm", str(wire), "100000", "1"])):
                        fault = json.loads(self.command(native.name + "-close-" + label, command))
                        self.assertEqual(fault["status"], 10)
                        self.assertGreater(fault["cleanup"], 0)
                        self.assertEqual(fault["unknown"], 1)
                        self.assertEqual((fault["opens"], fault["closes"], fault["fields"], fault["value"]), (1, 1, 99, 12345))
            if case == 1:
                self.tamper(compiler, flags, providers, ordinary, ldflags, generated, driver)
        malformed = self.artifacts / "truncated.nvm"
        malformed.write_bytes((self.artifacts / "case-0.nvm").read_bytes()[:-1])
        bad = json.loads(self.command("vm-truncated", [str(fixture), "vm", str(malformed), "100000", "0"]))
        self.assertNotEqual(bad["status"], 0)
        self.assertEqual((bad["acquired"], bad["opens"], bad["closes"], bad["fields"]), (0, 0, 0, 99))
        self.assertTrue(received4 and received6)
        self.assertTrue(all(byte == b"\xa5" for byte in received4 + received6))

    def validate(self, report, expected, case):
        self.assertEqual(report["status"], 9 if case == 3 else 0)
        self.assertEqual(report["cleanup"], 0)
        self.assertEqual(report["opens"], 0 if case == 2 else 1)
        self.assertEqual(report["closes"], report["opens"])
        if expected is None:
            self.assertEqual((report["fields"], report["value"]), (99, 12345))
        else:
            self.assertEqual((report["fields"], report["value"]), (1, expected))

    def tamper(self, compiler, flags, providers, ordinary, ldflags, code, driver):
        changes = {"abi": code.replace("indirect_native_abi(1u,", "indirect_native_abi(2u,")}
        for name, pattern in (("candidates", r"(call\.checked_candidates\)!=UINT64_C\()(\d+)"),
                              ("endpoint", r"(variant\.body\.pending_checks\)!=UINT64_C\()(\d+)")):
            count = 0
            def change(match):
                nonlocal count
                value = int(match[2])
                if not count and (name != "endpoint" or value & 512):
                    count += 1
                    return match[1] + str(value ^ (512 if name == "endpoint" else 1))
                return match[0]
            altered = re.sub(pattern, change, code)
            self.assertEqual(count, 1)
            changes[name] = altered
        for name, altered in changes.items():
            self.assertNotEqual(code, altered)
            source = self.artifacts / ("tamper-" + name + ".c")
            source.write_text(altered)
            exe = self.artifacts / ("tamper-" + name)
            self.command(exe.name + "-build", [*compiler, *flags, "-O1", str(source), str(driver),
                         *providers, *ordinary, *ldflags, "-o", str(exe)])
            report = json.loads(self.command(exe.name + "-run", [str(exe), "100000", "0"]))
            self.assertEqual(report["status"], 2)
            self.assertEqual((report["acquired"], report["opens"], report["closes"], report["fields"]), (0, 0, 0, 99))

if __name__ == "__main__":
    unittest.main()
