"""I reject missing or changed test runtimes before product tests start."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nano_wasmtime", ROOT / ".github/containers/instrumented/install_wasmtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class WasmtimePrerequisite(unittest.TestCase):
    def test_missing_runtime_has_actionable_diagnostic(self):
        with patch.object(runtime.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "I require wasmtime 50.0.1; run make install-test-wasmtime"):
                runtime.check()

    def test_installation_checks_identity_type_and_version_before_replacing_output(self):
        for failure in ("checksum", "link", "version", None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                destination = root / "bin"
                destination.mkdir()
                prior = destination / "wasmtime"
                prior.write_bytes(b"previous runtime")
                archive = root / "runtime.tar.xz"
                version = "0.0.0" if failure == "version" else runtime.VERSION
                body = f"#!/bin/sh\necho 'wasmtime {version} (fixture)'\n".encode()
                with tarfile.open(archive, "w:xz") as output:
                    member = tarfile.TarInfo("runtime/wasmtime")
                    if failure == "link":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "/bin/sh"
                        output.addfile(member)
                    else:
                        member.size = len(body)
                        output.addfile(member, io.BytesIO(body))
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                if failure == "checksum":
                    digest = "0" * 64
                if failure:
                    with self.assertRaises(ValueError):
                        runtime.install(archive, digest, "runtime/wasmtime", root)
                    self.assertEqual(prior.read_bytes(), b"previous runtime")
                else:
                    runtime.install(archive, digest, "runtime/wasmtime", root)
                    runtime.check(prior)
                self.assertEqual(list(destination.iterdir()), [prior])


if __name__ == "__main__":
    unittest.main()
