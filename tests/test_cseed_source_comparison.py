"""I refuse changed comparison inputs and preserve interleaved trace events."""
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from scripts import compare_cseed_source_creation as comparison


class CseedSourceComparison(unittest.TestCase):
    def test_different_and_missing_source_inputs_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory)/name for name in ('original', 'modified')]
            for root in roots:
                for name in (*comparison.FACTORY_FILES, 'src_nano/nanoc_v06.nano',
                             'modules/compiler_support/compiler_support.nano'):
                    path = root/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('# I retain this exact input.\n')
            self.assertTrue(comparison.require_same_inputs(*roots))
            path = roots[1]/'src_nano/nanoc_v06.nano'
            path.write_text('# I changed this input.\n')
            with self.assertRaisesRegex(ValueError, 'src_nano/nanoc_v06.nano'):
                comparison.require_same_inputs(*roots)
            path.write_text('# I retain this exact input.\n')
            (roots[1]/'modules/compiler_support/compiler_support.nano').unlink()
            with self.assertRaisesRegex(ValueError, 'modules/compiler_support/compiler_support.nano'):
                comparison.require_same_inputs(*roots)

    def test_interleaved_and_truncated_shadow_trace(self):
        root = Path('/worker/original')
        def event(kind, name, seconds=0):
            return json.dumps({'event': kind, 'name': name, 'source': str(root/'src_nano/test.nano'),
                               'line': 17, 'elapsed_seconds': seconds, 'failures': 0}, separators=(',', ':'))
        text = ('I retain an interleaved diagnostic.'+event('shadow-start', 'first')+'\n'+
                event('shadow-complete', 'first', .125)+'\n'+event('shadow-start', 'second'))
        result = comparison.shadow_trace(text, root)
        self.assertEqual((result['starts'], result['completes'], result['failures']), (2, 1, 0))
        self.assertEqual(result['summed_body_seconds'], .125)
        self.assertEqual(result['selection_prefix'], [('first', 'src_nano/test.nano', 17),
                                                     ('second', 'src_nano/test.nano', 17)])

    def test_successful_subset_cannot_replace_full_selection(self):
        full = [('first', 'source.nano', 1), ('second', 'source.nano', 2)]
        prefix, original = comparison.check_selection([], None, full, True)
        self.assertEqual(comparison.check_selection(prefix, original, full[:1], False), (full, full))
        with self.assertRaisesRegex(ValueError, 'successful shadow selection'):
            comparison.check_selection(prefix, original, full[:1], True)
        with self.assertRaisesRegex(ValueError, 'shadow order'):
            comparison.check_selection(prefix, original, list(reversed(full)), False)

        with self.assertRaisesRegex(ValueError, 'earlier observed prefix'):
            comparison.check_selection(full, None, full[:1], True)

    @unittest.skipUnless(sys.platform == 'darwin', 'I require macOS copy-on-write storage.')
    def test_complete_clone_keeps_baseline_links_modes_and_write_isolation(self):
        parent = Path.home()/'.worktrees/nanolang'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='comparison-fixture-', dir=parent) as directory:
            repository, target = Path(directory)/'repo', Path(directory)/'clone'
            repository.mkdir()
            env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=repository, env=env, stderr=subprocess.PIPE)
            git('init', '-q')
            git('config', 'user.name', 'NanoLang fixture')
            git('config', 'user.email', 'fixture@example.invalid')
            (repository/'changed.txt').write_text('I retain my baseline.\n')
            (repository/'shared.txt').write_text('I retain independent data.\n')
            (repository/'shared.txt').chmod(0o755)
            (repository/'alias').symlink_to('shared.txt')
            git('add', '.')
            git('commit', '-qm', 'baseline')
            baseline = git('rev-parse', 'HEAD').decode().strip()
            (repository/'changed.txt').write_text('I retain my candidate.\n')
            git('commit', '-qam', 'candidate')
            report = comparison.prepare_worktree(repository, target, baseline)
            self.assertEqual(report['tracked_entries'], 3)
            self.assertEqual((target/'changed.txt').read_text(), 'I retain my baseline.\n')
            self.assertEqual(os.readlink(target/'alias'), 'shared.txt')
            self.assertTrue((target/'shared.txt').stat().st_mode & 0o111)
            self.assertNotEqual((target/'shared.txt').stat().st_ino,
                                (repository/'shared.txt').stat().st_ino)
            subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=target, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (target/'shared.txt').write_text('I changed only my clone.\n')
            self.assertEqual((repository/'shared.txt').read_text(), 'I retain independent data.\n')
