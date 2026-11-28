"""I relocate actual native imports and refuse changed or missing provider bytes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import time
import unittest

from scripts.install_host_artifacts import install, uninstall, declared_artifacts, section_bytes
from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]
CC = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', os.environ.get('CC', 'cc')))


class HostArtifactDeployment(unittest.TestCase):
    def setUp(self):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix='host-artifact-deployment-', dir=ROOT/'.test_output'))
        self.commands = []
        print(f'I retain host deployment products at {self.work}', flush=True)

    def command(self, argv, success=True, **kwargs):
        started = time.monotonic()
        result = run_captured(list(map(str, argv)), timeout=90, cwd=self.work, text=True, **kwargs)
        self.commands.append(dict(argv=list(map(str, argv)), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, elapsed_seconds=time.monotonic()-started))
        (self.work/'commands.json').write_text(json.dumps(self.commands, indent=2)+'\n')
        self.assertEqual(result.returncode == 0, success, result.stdout+result.stderr)
        return result

    def product(self):
        source = self.work/'provider.c'
        source.write_text('const char *path_basename(const char *s) { (void)s; return "exact-host-ok"; }\n')
        library = self.work/('same provider.dylib' if sys.platform == 'darwin' else 'same provider.so')
        self.command([*CC, '-shared', '-fPIC', source, '-o', library])
        assembly = self.work/'input.nasm'
        assembly.write_text(f'.import {json.dumps(str(library))} "path_basename" string string\n'
                            '.import_kind 0 artifact\n.string input "ignored"\n.entry main\n'
                            '.function main 0 0 0 int 1\nPUSH_STR input\nCALL_EXTERN 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n')
        module = self.work/'input.nvm'
        self.command([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        self.command([ROOT/'bin/nano_vm', '--verify-only', module])
        generated = self.work/'generated.c'
        self.command([ROOT/'bin/nvm2c', module, '-o', generated])
        binary = self.work/'product'
        self.command([*CC, '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', generated,
                      '-o', binary, '-lm', *(['-ldl'] if sys.platform.startswith('linux') else [])])
        self.assertEqual(self.command([binary]).stdout, 'exact-host-ok\n')
        return binary, library, module

    def test_relocated_native_uses_only_exact_packaged_provider(self):
        binary, library, module = self.product()
        original_module = module.read_bytes()
        prefix = self.work/'first prefix'
        (prefix/'bin').mkdir(parents=True)
        shutil.copy2(binary, prefix/'bin/product')
        receipt = install(binary, prefix, 'product')
        self.assertEqual(len(receipt['artifacts']), 1)
        key = receipt['artifacts'][0]['sha256']
        self.assertEqual(key, hashlib.sha256(library.read_bytes()).hexdigest())
        # I relocate the complete installation and remove the source artifact.
        relocated = self.work/'relocated prefix'
        prefix.rename(relocated)
        library.rename(self.work/'retained-original-provider')
        installed = relocated/'bin/product'
        self.assertEqual(self.command([installed]).stdout, 'exact-host-ok\n')
        packaged = relocated/'lib/nanolang/hosts'/key
        valid = packaged.read_bytes()
        # I reject different bytes at the expected content-addressed path.
        packaged.write_bytes(valid+b'changed')
        self.command([installed], success=False)
        packaged.write_bytes(valid)
        self.assertEqual(self.command([installed]).stdout, 'exact-host-ok\n')
        # A same-basename provider cannot fill a missing content-addressed entry.
        packaged.rename(packaged.with_name(library.name))
        self.command([installed], success=False)
        self.assertEqual(module.read_bytes(), original_module)

    def test_development_and_installer_refuse_changed_original(self):
        binary, library, _ = self.product()
        original = library.read_bytes()
        library.write_bytes(original+b'changed')
        self.command([binary], success=False)
        prefix = self.work/'refused-prefix'
        with self.assertRaisesRegex(ValueError, 'changed host-artifact bytes'):
            install(binary, prefix, 'product')
        self.assertFalse(prefix.exists())
        library.write_bytes(original)
        self.assertEqual(self.command([binary]).stdout, 'exact-host-ok\n')
        first = install(binary, prefix, 'first')
        (prefix/'bin').mkdir()
        shutil.copy2(binary, prefix/'bin/product')
        packaged = prefix/'lib/nanolang/hosts'/first['artifacts'][0]['sha256']
        packaged.write_bytes(original+b'changed')
        self.command([prefix/'bin/product'], success=False)
        packaged.write_bytes(original)
        self.assertEqual(self.command([prefix/'bin/product']).stdout, 'exact-host-ok\n')
        install(binary, prefix, 'second')
        uninstall(prefix, 'first')
        self.assertEqual(len(list((prefix/'lib/nanolang/hosts').iterdir())), 1)
        uninstall(prefix, 'second')
        self.assertEqual(list((prefix/'lib/nanolang/hosts').iterdir()), [])

    def test_missing_import_remains_lazy_but_cannot_be_deployed_unbound(self):
        _, library, module = self.product()
        library.unlink()
        source = self.work/'missing.c'
        self.command([ROOT/'bin/nvm2c', module, '-o', source])
        binary = self.work/'missing'
        self.command([*CC, '-std=c11', '-O1', source, '-o', binary, '-lm',
                      *(['-ldl'] if sys.platform.startswith('linux') else [])])
        self.command([binary], success=False)
        with self.assertRaisesRegex(ValueError, 'unbound native host artifact'):
            install(binary, self.work/'refused-prefix', 'missing')

    def test_ordinary_strings_cannot_declare_host_artifacts(self):
        source = self.work/'ordinary.c'
        source.write_text('#include <stdio.h>\nint main(void) { puts("NANOLANG_HOST_V1:not-a-dependency"); return 0; }\n')
        binary = self.work/'ordinary'
        self.command([*CC, source, '-o', binary])
        self.assertEqual(install(binary, self.work/'prefix', 'ordinary')['artifacts'], [])
        with self.assertRaises(ValueError):
            declared_artifacts(binary.read_bytes()[:48])
        declared, _, _ = self.product()
        image = declared.read_bytes()
        record = next(record for section in section_bytes(image) for record in section.split(b'\0') if record)
        marker = image.index(record)
        changed = image[:marker]+b'X'+image[marker+1:]
        # My section parser refuses a corrupted record rather than ignoring it.
        with self.assertRaisesRegex(ValueError, 'malformed native host-artifact'):
            declared_artifacts(changed)

    def test_digest_matches_independent_standard_and_boundary_vectors(self):
        source = self.work/'digest.c'
        source.write_text('#include "'+str(ROOT/'src/nanoisa/host_artifact_digest.h')+'"\n'
                          'int main(int argc,char **argv) { char digest[65]; '
                          'if(argc!=2||!nano_host_digest_file(argv[1],digest)) return 1; puts(digest); return 0; }\n')
        binary = self.work/'digest'
        self.command([*CC, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', source, '-o', binary])
        vectors = [b'', b'abc', b'a'*1000000]+[bytes(range(256))[:n] for n in (55,56,63,64,65,119,120,127,128,129,255)]
        vectors += [bytes(range(256))*65]
        for data in vectors:
            with self.subTest(length=len(data)):
                path = self.work/'bytes';path.write_bytes(data)
                self.assertEqual(self.command([binary, path]).stdout.strip(), hashlib.sha256(data).hexdigest())
        self.command([binary, self.work/'missing'], success=False)
