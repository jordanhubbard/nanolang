"""I refuse changed comparison inputs and preserve interleaved trace events."""
import json
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
