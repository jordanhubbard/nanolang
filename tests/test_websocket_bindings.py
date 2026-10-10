"""I load my packaged WebSocket module through native and bytecode drivers."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WebSocketBindings(unittest.TestCase):
    def test_packaged_refusals_and_dependency_shadows(self):
        with tempfile.TemporaryDirectory(prefix="nano-websocket-bindings-") as tmp:
            work = Path(tmp)
            env = os.environ.copy()
            env["NANO_BUILD_CACHE"] = str(work / "cache")
            source = ROOT / "tests/websocket_safety.nano"
            commands = (
                [ROOT / "bin/nanoc_c", source, "-o", work / "native"],
                [work / "native"],
                [ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", work / "program.nvm"],
                [ROOT / "bin/nano_vm", work / "program.nvm"],
            )
            for command in commands:
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, f"{command}\n{result.stdout}{result.stderr}")


if __name__ == "__main__":
    unittest.main()
