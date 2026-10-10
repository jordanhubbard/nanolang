"""I execute independently serialized File bytes from both compiler producers."""
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from tests import test_service_lowering as lowering

ROOT = Path(__file__).resolve().parents[1]


class ServiceWire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='nano-service-wire-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name)
        cls.compiler = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc'))
        cls.command = staticmethod(lowering.ServiceLowering.command)
        producers = [[ROOT/'bin/nano_virt']]
        driver = os.environ.get('NANO_SERVICE_WIRE_DRIVER_MODULE')
        if driver:
            producers.append([ROOT/'bin/nano_vm', driver, '--'])
        cls.outputs = []
        for index, producer in enumerate(producers):
            module = cls.work/f'writer-{index}.nvm'
            cls.command([*producer, ROOT/'tests/service_wire.nano', '--emit-nvm', '-o', module])
            generated = module.with_suffix('.c')
            native = module.with_suffix('.native')
            cls.command([ROOT/'bin/nvm2c', module, '-o', generated])
            cls.command([*cls.compiler, '-std=c11', '-O2', '-g', '-fsanitize=address,undefined',
                         '-fno-sanitize-recover=all', generated, '-o', native, '-lm',
                         *(['-ldl'] if sys.platform.startswith('linux') else [])])
            for command in ([ROOT/'bin/nano_vm', module], [native]):
                text = cls.command(command)
                cls.outputs.append(bytes(map(int, text.split())))

    def test_independent_bytes_execute_with_explicit_grants(self):
        wire = self.outputs[0]
        for other in self.outputs[1:]:
            self.assertEqual(wire, other)
        self.assertEqual(wire[:4], b'NVM\x02')
        self.assertIn('entry_é'.encode('utf-8'), wire)
        self.assertEqual(struct.unpack_from('<I', wire, 36)[0], zlib.crc32(wire[40:]))
        self.assertEqual(struct.unpack_from('<Q', wire, 12)[0], len(wire))
        self.assertEqual(struct.unpack_from('<I', wire, 24)[0], 11)
        module = self.work/'file.nvm'
        module.write_bytes(wire)
        run = subprocess.run([ROOT/'bin/nano_vm', '--allow-temporary-files', '--file-cyclic',
                              '--file-instruction-limit', '100000', module], capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 42, run.stderr)
        generated = self.work/'source.c'
        native = self.work/'source.native'
        self.command([ROOT/'bin/nvm2c', '--file-temporary', '--file-cyclic', '--entry-name',
                      'source', module, '-o', generated])
        (self.work/'nanolang').mkdir()
        (self.work/'nanolang/file').symlink_to(ROOT/'src', target_is_directory=True)
        wrapper = self.work/'wrapper.c'
        wrapper.write_text(lowering.WRAPPER)
        self.command([*self.compiler, '-std=c11', '-O2', '-g', '-fsanitize=address,undefined',
                      '-fno-sanitize-recover=all', '-I'+str(self.work), '-Isrc', generated,
                      wrapper, ROOT/'lib/libnano_file_runtime.a', '-o', native, '-lm'])
        self.assertIn('EXEC 0 VALUE 42', self.command([native]))

    def test_ungranted_and_damaged_modules_refuse(self):
        for damaged in (False, True):
            wire = bytearray(self.outputs[0])
            if damaged:
                wire[-1] ^= 1
            module = self.work/f'refused-{damaged}.nvm'
            module.write_bytes(wire)
            options = ['--allow-temporary-files', '--file-cyclic', '--file-instruction-limit', '100000'] if damaged else []
            run = subprocess.run([ROOT/'bin/nano_vm', *options, module], capture_output=True, timeout=30)
            self.assertNotEqual(run.returncode, 0, run.stderr)
            self.assertNotEqual(run.returncode, 42, run.stderr)
            output = self.work/'prior.c'
            output.write_bytes(b'prior-output')
            run = subprocess.run([ROOT/'bin/nvm2c', *(['--file-temporary', '--file-cyclic', '--entry-name', 'source'] if damaged else []),
                                  module, '-o', output], capture_output=True, timeout=30)
            self.assertNotEqual(run.returncode, 0, run.stderr)
            self.assertEqual(output.read_bytes(), b'prior-output')
