"""I rebuild my real C driver twice without rewriting shared compiler providers."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]


class CompilerVersion(unittest.TestCase):
    def command(self, args, *, build=False, env=None):
        result = run_captured(args, cwd=ROOT, env=env, text=True,
                              timeout=30, build=build)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_real_driver_rebuilds_have_identical_version_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-version-builds-") as temporary:
            directory = Path(temporary)
            makefile = directory / "Makefile"
            makefile.write_text("include Makefile.gnu\n"
                                ".PHONY: version-providers\n"
                                "version-providers:\n\t@printf '%s\\n' $(COMPILER_OBJECTS)\n")
            providers = self.command(["make", "--no-print-directory", "-s", "-f",
                                      makefile, "version-providers"]).splitlines()
            self.assertIn("obj/main.o", providers)
            providers.remove("obj/main.o")

            def snapshot():
                return {str(path): (path.stat().st_mtime_ns,
                                   hashlib.sha256(path.read_bytes()).hexdigest())
                        for path in [ROOT / "bin/nanoc_c", *(ROOT / p for p in providers)]}

            before = snapshot()
            reports = []
            for index, epoch in enumerate(("946684800", "978307200")):
                private_main = directory / f"main-{index}.o"
                compiler = directory / f"nanoc-{index}"
                makefile.write_text("include Makefile.gnu\n"
                                    f"{private_main}: src/main.c $(HEADERS) src/version.h\n"
                                    "\t$(CC) $(CFLAGS) $(DEPFLAGS) -c src/main.c -o $@\n")
                env = dict(os.environ, SOURCE_DATE_EPOCH=epoch)
                command = ["make", "--no-print-directory", "-f", makefile,
                           f"COMPILER_C={compiler}",
                           "COMPILER_OBJECTS=" + " ".join([str(private_main), *providers])]
                for provider in providers:
                    command += ["-o", provider]
                self.command([*command, compiler], build=True, env=env)
                report = self.command([compiler, "--version"], env=env)
                self.assertEqual(self.command([compiler, "-v"], env=env), report)
                reports.append(report)
                self.assertEqual(snapshot(), before, "I must preserve shared compiler inputs and outputs.")
            self.assertEqual(reports[0], reports[1])
            version = json.loads((ROOT / "package.json").read_text())["version"]
            self.assertEqual(reports[0].splitlines(), ["nanoc " + version, "nanolang compiler"])


if __name__ == "__main__":
    unittest.main()
