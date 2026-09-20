"""I check raw CAST_U8; source lowering and other backends remain separate."""
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_cyclic


class CastU8(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-cast-u8-'))
        print(f'I retain raw byte conversion artifacts at {cls.artifacts}', flush=True)
        cls.cc = shlex.split(os.environ.get('CAST_U8_CC', 'cc'))
        cls.flags = shlex.split(os.environ.get('CAST_U8_CFLAGS', ''))
        cls.objects = shlex.split(os.environ['CAST_U8_OBJECTS'])
        cls.ldflags = shlex.split(os.environ.get('CAST_U8_LDFLAGS', ''))
        (cls.artifacts / 'selection.json').write_text(json.dumps({
            'cc': cls.cc, 'flags': cls.flags, 'objects': cls.objects,
            'ldflags': cls.ldflags,
            'instrumentation': 'I rebuild vm.c and my fixture with selected flags; linked providers retain their inventoried build flags.',
        }, indent=2) + '\n')

    def test_exact_bytes_both_dispatches(self):
        for dispatch, flags in [('switch', ['-DNANO_NO_COMPUTED_GOTO']), ('default', [])]:
            executable = self.artifacts / dispatch
            self.command(dispatch + '-build', [*self.cc, *self.flags,
                '-std=gnu11', '-UNDEBUG', '-I./src', *flags,
                'tests/nanovm/test_cast_u8.c', 'src/nanovm/vm.c',
                *self.objects, *self.ldflags, '-o', str(executable)])
            output = self.command(dispatch + '-run', [str(executable)])
            self.assertIn(b'raw byte conversion assertions.', output)


if __name__ == '__main__':
    unittest.main()
