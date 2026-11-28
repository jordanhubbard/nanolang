"""I restore only uniquely evidenced cache pointers before native diagnostics."""
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import ci_darwin_shadow_lifecycle as diagnostic


class CacheNamespaceRestore(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix='nano-diagnostic-namespace-')
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.out = self.root / '.test_output/observer'

    def cache(self, relative='modules/std/.build', *, generation='AbC123'):
        cache = self.root / relative
        target = cache / ('.nano-gen-' + generation)
        target.mkdir(parents=True)
        for name, data in [('libstd.dylib', b'native-library\0'), ('std.o', b'native-object'),
                           ('.hash_cache', b'{"identity":1}\n')]:
            (target / name).write_bytes(data)
        shutil.copytree(target, cache / 'current')
        return cache, target

    def test_atomic_publisher_and_unique_restoration(self):
        cache, target = self.cache()
        original = diagnostic.cache_generation_files(cache / 'current')
        previous_target = diagnostic.cache_generation_files(target)
        probe = cache / '.publication-probe'
        probe.symlink_to(target.name, target_is_directory=True)
        with self.assertRaises(OSError) as failed:
            os.replace(probe, cache / 'current')
        self.assertIn(failed.exception.errno, (errno.EISDIR, errno.ENOTDIR))
        report = diagnostic.restore_cache_currents(self.root, self.out)
        self.assertTrue(report['complete'])
        self.assertFalse(report['original_link_metadata_available'])
        self.assertEqual(len(report['entries']), 1)
        row = report['entries'][0]
        self.assertTrue(row['restored'])
        self.assertEqual(os.readlink(cache / 'current'), target.name)
        self.assertEqual(diagnostic.cache_generation_files(self.root / row['backup']), original)
        self.assertEqual(diagnostic.cache_generation_files(target), previous_target)
        os.replace(probe, cache / 'current')
        self.assertEqual(os.readlink(cache / 'current'), target.name)
        self.assertEqual(diagnostic.cache_generation_files((cache / 'current').resolve()), original)

    def test_both_cache_layouts_and_unrelated_directory(self):
        local, first = self.cache()
        global_, second = self.cache('obj/module_cache/v2-' + 'a' * 64, generation='DeF456')
        unrelated = self.root / 'modules/assets/current'
        unrelated.mkdir(parents=True)
        (unrelated / 'source').write_text('untouched')
        report = diagnostic.restore_cache_currents(self.root, self.out)
        self.assertEqual(len(report['entries']), 2)
        self.assertEqual(os.readlink(local / 'current'), first.name)
        self.assertEqual(os.readlink(global_ / 'current'), second.name)
        self.assertFalse(unrelated.is_symlink())
        self.assertEqual((unrelated / 'source').read_text(), 'untouched')

    def test_second_pass_keeps_pointers_generations_and_flattened_evidence(self):
        cache, target = self.cache()
        first = diagnostic.restore_cache_currents(self.root, self.out)
        before = diagnostic.cache_generation_files(target)
        backup = self.root / first['entries'][0]['backup']
        second = diagnostic.restore_cache_currents(self.root, self.out)
        self.assertTrue(second['complete'])
        self.assertFalse(second['entries'][0]['restored'])
        self.assertEqual(second['entries'][0]['basis'], 'existing-safe-relative-link')
        self.assertEqual(os.readlink(cache / 'current'), target.name)
        self.assertEqual(diagnostic.cache_generation_files(target), before)
        self.assertEqual(diagnostic.cache_generation_files(backup), before)

    def test_ambiguous_generations_refuse_without_cache_mutation(self):
        cache, target = self.cache()
        shutil.copytree(target, cache / '.nano-gen-GhI789')
        before = diagnostic.cache_generation_files(cache / 'current')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            diagnostic.restore_cache_currents(self.root, self.out)
        self.assertFalse((cache / 'current').is_symlink())
        self.assertEqual(diagnostic.cache_generation_files(cache / 'current'), before)

    def test_missing_changed_and_extra_generation_contents_refuse(self):
        for mutation in ('missing', 'changed', 'extra'):
            with self.subTest(mutation=mutation):
                cache, target = self.cache('modules/' + mutation + '/.build')
                if mutation == 'missing':
                    (target / 'std.o').unlink()
                elif mutation == 'changed':
                    (target / 'std.o').write_bytes(b'another-object')
                else:
                    (target / 'extra').write_bytes(b'unrecorded')
                with self.assertRaisesRegex(ValueError, 'found 0'):
                    diagnostic.restore_cache_currents(self.root, self.out)
                self.assertFalse((cache / 'current').is_symlink())
                shutil.rmtree(cache.parent)

    def test_all_plans_validate_before_first_cache_change(self):
        local, _ = self.cache()
        other, target = self.cache('obj/module_cache/v2-' + 'b' * 64)
        shutil.copytree(target, other / '.nano-gen-GhI789')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            diagnostic.restore_cache_currents(self.root, self.out)
        self.assertFalse((local / 'current').is_symlink())
        self.assertFalse((other / 'current').is_symlink())
        self.assertFalse((self.out / 'flattened-cache-currents').exists())

    def test_failed_link_creation_restores_flattened_directory(self):
        cache, target = self.cache()
        before = diagnostic.cache_generation_files(cache / 'current')
        with patch.object(Path, 'symlink_to', side_effect=OSError('injected link failure')):
            with self.assertRaisesRegex(OSError, 'injected'):
                diagnostic.restore_cache_currents(self.root, self.out)
        self.assertFalse((cache / 'current').is_symlink())
        self.assertEqual(diagnostic.cache_generation_files(cache / 'current'), before)
        self.assertEqual(diagnostic.cache_generation_files(target), before)
        report = json.loads((self.out / 'cache-current-restoration.json').read_text())
        self.assertFalse(report['complete'])
        self.assertFalse(report['entries'][0]['restored'])
        self.assertIn('injected', report['exception'])

    def test_empty_and_nested_current_trees_refuse(self):
        cache, _ = self.cache()
        shutil.rmtree(cache / 'current')
        (cache / 'current').mkdir()
        with self.assertRaisesRegex(ValueError, 'empty'):
            diagnostic.restore_cache_currents(self.root, self.out)
        (cache / 'current/nested').mkdir()
        with self.assertRaisesRegex(ValueError, 'regular'):
            diagnostic.restore_cache_currents(self.root, self.out)

    def test_nonregular_generation_entry_refuses(self):
        cache, target = self.cache()
        (target / 'external').symlink_to(self.root / 'outside')
        with self.assertRaisesRegex(ValueError, 'regular'):
            diagnostic.restore_cache_currents(self.root, self.out)
        self.assertFalse((cache / 'current').is_symlink())

    def test_current_regular_file_refuses(self):
        cache, _ = self.cache()
        shutil.rmtree(cache / 'current')
        (cache / 'current').write_bytes(b'not a directory')
        with self.assertRaisesRegex(ValueError, 'real cache generation'):
            diagnostic.restore_cache_currents(self.root, self.out)
        self.assertEqual((cache / 'current').read_bytes(), b'not a directory')

    def test_escaped_and_dangling_current_links_refuse(self):
        cache, _ = self.cache()
        shutil.rmtree(cache / 'current')
        for target in ('../.nano-gen-AbC123', '/tmp/elsewhere', '.nano-gen-XyZ789'):
            with self.subTest(target=target):
                (cache / 'current').symlink_to(target, target_is_directory=True)
                with self.assertRaises(ValueError):
                    diagnostic.restore_cache_currents(self.root, self.out)
                self.assertEqual(os.readlink(cache / 'current'), target)
                (cache / 'current').unlink()

    def test_evidence_and_cache_roots_cannot_escape(self):
        self.cache()
        with tempfile.TemporaryDirectory(prefix='nano-outside-namespace-') as outside:
            with self.assertRaisesRegex(ValueError, 'inside my private root'):
                diagnostic.restore_cache_currents(self.root, Path(outside))
            shutil.rmtree(self.root / 'modules')
            (self.root / 'modules').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'escaped cache root'):
                diagnostic.restore_cache_currents(self.root, self.out)


if __name__ == '__main__':
    unittest.main()
