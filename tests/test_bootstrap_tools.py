"""I keep unchanged tool binaries stable while rebuilding changed or missing tools."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BootstrapTools(unittest.TestCase):
    def test_real_make_links_only_changed_or_missing_tool(self):
        with tempfile.TemporaryDirectory(prefix='nano-bootstrap-tools-') as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / 'Makefile.gnu', root / 'Makefile.gnu')
            objects = ['obj/probe.o', 'obj/nanovirt/main.o', 'obj/nanovm/main.o',
                       'obj/nanovm/vmd_protocol.o', 'obj/nanovm/vmd_client.o']
            binaries = ['bin/nano_virt', 'bin/nano_vm', 'bin/nanoisa', 'bin/nvm2c']
            for name in objects + binaries:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                os.utime(path, (100 if name in objects else 200,) * 2)
            command = [os.environ.get('MAKE_BIN', 'make'), '-f', 'Makefile.gnu', '-n',
                       'UNAME_S=Linux', 'NANOISA_OBJECTS=', 'NANOVIRT_OBJECTS=',
                       'NANOVM_OBJECTS=', 'COMMON_OBJECTS=obj/probe.o', 'RUNTIME_OBJECTS=', 'NANOISA_UTF8=obj/probe.o',
                       'NANOISA_DUMP_OBJECT=', 'NVM2C_MAIN_OBJECT=', 'FILE_CLI_OBJECT=',
                       'FILE_PUBLIC_LIBRARY=', 'SERVICE_DRIVER_OBJECTS=', '-o', 'check-binary64-parser',
                       '-o', 'bin/nano_as_capture.so']
            for name in objects[1:]:
                command += ['-o', name]
            command += ['nano_virt', 'nano_vm', 'nanoisa_dump', 'nvm2c']

            def links():
                result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return [name for name in binaries if '-o ' + name + ' ' in result.stdout]

            self.assertEqual(links(), [])
            os.utime(root / 'obj/probe.o', (300, 300))
            self.assertEqual(links(), binaries)
            os.utime(root / 'obj/probe.o', (100, 100))
            for name in binaries:
                with self.subTest(missing=name):
                    (root / name).unlink()
                    self.assertEqual(links(), [name])
                    (root / name).touch()
                    os.utime(root / name, (200, 200))


if __name__ == '__main__':
    unittest.main()
