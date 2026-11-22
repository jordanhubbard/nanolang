"""I refuse unknown selection metadata without exporting a replacement tool."""
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from scripts import ci_select_capture_assembler as selector


@unittest.skipUnless(os.name == 'posix', 'I exercise the POSIX CI selector.')
class SelectCaptureAssembler(unittest.TestCase):
    def test_unknown_family_never_builds_or_exports_a_profile(self):
        self.refusal('2.47', 'unknown selected assembler family')

    def test_missing_supplied_profile_never_repeats_its_producer(self):
        self.refusal('2.46', 'supplied completed assembler profile', provided=True)

    def refusal(self, version, message, provided=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'modules/compiler_support').mkdir(parents=True)
            (root / 'obj').mkdir()
            (root / 'src').mkdir()
            (root / 'src/module_builder.c').write_text('CI metadata refusal fixture\n')
            assembler, compiler = root / 'unknown-as', root / 'selected-cc'
            assembler.write_text("#!/bin/sh\nprintf 'GNU assembler (fixture) " + version + "\\n'\n")
            compiler.write_text('#!/bin/sh\nprintf "%s\\n" ' + shlex.quote(str(assembler)) + '\n')
            probe = root / 'obj/test_module_generation_probe'
            probe.write_text('#!/bin/sh\nexit 1\n')
            for path in (assembler, compiler, probe):
                path.chmod(0o700)
            (root / 'modules/compiler_support/module.json').write_text(json.dumps({'c_compiler': str(compiler)}))
            exported = root / 'environment'
            exported.write_text('retained=original\n')
            with patch.object(selector, 'ROOT', root), patch.object(selector.process, 'ROOT', root), \
                 patch.object(selector.diagnostics, 'ROOT', root), patch.object(selector.sys, 'platform', 'linux'), \
                 patch.dict(os.environ, {'GITHUB_ENV': str(exported),
                     'NANO_CI_FIXED_ASSEMBLER_DIRECTORY': str(root / 'absent-profile') if provided else ''},
                     clear=False):
                with patch.dict(os.environ):
                    os.environ.pop('NANO_CC', None)
                    with self.assertRaisesRegex(RuntimeError, message):
                        selector.main()
            self.assertEqual(exported.read_text(), 'retained=original\n')
            self.assertFalse((root / '.test_output/source-capture-diagnostic/fixed-as246').exists())
            result = json.loads((root / '.test_output/capture-assembler-ci/result.json').read_text())
            self.assertNotIn('selected_assembler', result)
            self.assertEqual(result['compiler_before']['configured'], str(compiler))

    def test_ambiguous_compiler_command_is_preserved_and_refused(self):
        with patch.dict(os.environ, NANO_CC='cc -unexpected-ci-flag'):
            with self.assertRaisesRegex(RuntimeError, 'one literal configured CI compiler'):
                selector.compiler_identity()
            self.assertEqual(os.environ['NANO_CC'], 'cc -unexpected-ci-flag')


if __name__ == '__main__':
    unittest.main()
