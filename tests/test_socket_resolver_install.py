"""I select and execute my installed resolver without PATH or CWD lookup."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ResolverInstall(unittest.TestCase):
    def test_installed_relocated_helper_and_explicit_selection(self):
        with tempfile.TemporaryDirectory(prefix="nano resolver install ") as tmp:
            work = Path(tmp)
            prefix = work / "original prefix"
            cc = shlex.split(os.environ.get("CC", "cc"))
            subprocess.run(["make", "-f", "Makefile.gnu", "install-resolver", f"PREFIX={prefix}",
                            "CC=" + shlex.join(cc)], cwd=ROOT, check=True, capture_output=True, timeout=60)
            probe = prefix / "bin/probe"
            command = cc + ["-std=c11", "-D_DEFAULT_SOURCE", "-D_DARWIN_C_SOURCE", "-Wall", "-Wextra", "-Werror"]
            command += [str(ROOT / p) for p in ("tests/socket_resolver_path_probe.c", "src/nsi_socket_resolver.c", "src/nsi_socket.c", "src/nsi_cap.c")]
            subprocess.run(command + ["-o", str(probe)], check=True, capture_output=True, timeout=60)
            env = {k:v for k,v in os.environ.items() if k not in ("NANOLANG_RESOLVER", "NANOLANG_ROOT")}
            def run(expected, **updates):
                result = subprocess.run([str(probe)], cwd=work, env=dict(env, **updates), text=True,
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                return result.stdout.strip()
            self.assertEqual(run(0), str((prefix / "bin/nano-resolver").resolve()))
            moved = work / "relocated prefix"
            prefix.rename(moved)
            probe = moved / "bin/probe"
            helper = moved / "bin/nano-resolver"
            self.assertEqual(run(0), str(helper.resolve()))
            self.assertEqual(run(0, NANOLANG_RESOLVER=str(helper)), str(helper.resolve()))
            self.assertEqual(run(0, NANOLANG_ROOT=str(moved)), str(helper.resolve()))
            for setting in ("", "relative", str(work / "missing"), str(work)):
                run(1, NANOLANG_RESOLVER=setting)
            for setting in ("", "relative", str(work / "missing")):
                run(1, NANOLANG_ROOT=setting)
            isolated = work / "isolated"
            isolated.mkdir()
            shutil.copy2(probe, isolated / "probe")
            probe = isolated / "probe"
            shutil.copy2(helper, work / "nano-resolver")
            run(1, PATH=str(helper.parent))
            run(0, NANOLANG_RESOLVER=str(helper))

if __name__ == "__main__":
    unittest.main()
