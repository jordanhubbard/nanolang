"""I keep my public C version and release metadata in agreement."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseVersion(unittest.TestCase):
    def compile_header_version(self, header, directory):
        source = directory / "version.c"
        source.write_text('''#include <stdio.h>
#include "version.h"
int main(void) {
    printf("%s\\n%d.%d.%d%s\\n", NANOLANG_VERSION,
           NANOLANG_VERSION_MAJOR, NANOLANG_VERSION_MINOR,
           NANOLANG_VERSION_PATCH, NANOLANG_VERSION_SUFFIX);
    return 0;
}
''')
        binary = directory / "version"
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                       ["-I", str(header.parent), str(source), "-o", str(binary)],
                       check=True, capture_output=True)
        return subprocess.check_output([str(binary)], text=True).splitlines()

    def test_future_release_updates_both_outputs(self):
        for version in ("12.34.56", "12.34.56-rc.1"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                package = directory / "package.json"
                header = directory / "version.h"
                subprocess.run(["python3", str(ROOT / "scripts/generate_root_package_json.py"),
                                version, "--output", str(package),
                                "--version-header", str(header)], check=True)
                self.assertEqual(json.loads(package.read_text())["version"], version)
                self.assertEqual(self.compile_header_version(header, directory), [version, version])
                before = (package.read_bytes(), header.read_bytes())
                result = subprocess.run(["python3", str(ROOT / "scripts/generate_root_package_json.py"),
                                         "invalid", "--output", str(package),
                                         "--version-header", str(header)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((package.read_bytes(), header.read_bytes()), before)

    def test_current_header_and_compiler_match_release(self):
        version = json.loads((ROOT / "package.json").read_text())["version"]
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(self.compile_header_version(ROOT / "src/version.h", Path(temp)),
                             [version, version])
        result = subprocess.run([str(ROOT / "bin/nanoc_c"), "--version"],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.splitlines()[0], "nanoc " + version)


if __name__ == "__main__":
    unittest.main()
