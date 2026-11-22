"""I refuse incomplete preceding capture and production source drift."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import ci_qualify_fixed_userguide as guide


class FixedUserguideQualification(unittest.TestCase):
    def test_actual_retained_cold_record_resolves_external_and_relative_tool_keys(self):
        fixture = Path(__file__).parent / 'fixtures/capture-cold-tool-keys.json'
        record = json.loads(fixture.read_text())
        self.assertEqual(guide.retained_tool_digest(record, record['compiler']),
            '864711037b900acdb7a449cf48d912cefa935b527fb60c767ccb956779e66136')
        self.assertEqual(guide.retained_tool_digest(record, record['assembler']),
            'f881e708d6cd94ee8d3d1b9d163cfd502f5447b0436887e4d1e66b7c3134e310')

    def test_unknown_retained_tool_or_original_probe_root_is_refused(self):
        fixture = Path(__file__).parent / 'fixtures/capture-cold-tool-keys.json'
        record = json.loads(fixture.read_text())
        for selected in ('/usr/bin/unknown-tool', record['assembler'] + '-unknown'):
            with self.subTest(selected=selected), self.assertRaisesRegex(RuntimeError, 'retained tool identity'):
                guide.retained_tool_digest(record, selected)
        record['command'][0] = '/unrecognized/other-probe'
        with self.assertRaisesRegex(RuntimeError, 'original capture probe root'):
            guide.retained_tool_digest(record, record['assembler'])

    def fixture(self, root):
        source = dict(discovered=128, ran=128, errors=0, failures=0, skipped=[])
        response = dict(discovered=1, ran=1, errors=0, failures=0, skipped=[])
        result = dict(fixed_as246_complete=True, fixed_identified_profile_qualification_passed=True,
            checkout_sources_restored=True, complete_corpus=dict(source=source, response=response,
                separate_invocations=True, total_methods=129))
        for name, value in (('tests.json', source), ('response-tests.json', response), ('result.json', result)):
            (root / name).write_text(json.dumps(value))
        (root / 'cold-compiler-support').mkdir()
        (root / 'cold-compiler-support/observation.json').write_text(
            json.dumps(dict(returncode=0, supported_version_decision=0)))
        return result

    def test_incomplete_original_corpus_cannot_authorize_userguide_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertTrue(guide.prior_reports(root)[0]['fixed_as246_complete'])
            path = root / 'tests.json'
            incomplete = json.loads(path.read_text()); incomplete['ran'] = 127
            path.write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(RuntimeError, 'complete passing original capture reports'):
                guide.prior_reports(root)

    def test_mismatched_embedded_report_and_failed_cold_publication_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.fixture(root)
            changed = copy.deepcopy(result); changed['complete_corpus']['source']['skipped'] = [['missing', 'bad']]
            (root / 'result.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, 'matching private fixed capture cohort'):
                guide.prior_reports(root)
            (root / 'result.json').write_text(json.dumps(result))
            (root / 'cold-compiler-support/observation.json').write_text(
                json.dumps(dict(returncode=1, supported_version_decision=0)))
            with self.assertRaisesRegex(RuntimeError, 'preceding cold publication'):
                guide.prior_reports(root)

    def test_real_git_source_drift_is_refused_after_diagnostic_only_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', '-c', 'user.name=CaptureControl', '-c',
                    'user.email=capture-control@example.invalid', *args], cwd=root, stderr=subprocess.PIPE).decode().strip()
            git('init', '-q')
            (root / 'src').mkdir(); (root / 'docs').mkdir(); (root / 'output').mkdir()
            source = root / 'src/module_builder.c'; source.write_text('original production source\n')
            git('add', 'src'); git('commit', '-qm', 'original'); pin = git('rev-parse', 'HEAD')
            (root / 'docs/ROADMAP.md').write_text('diagnostic evidence\n')
            git('add', 'docs'); git('commit', '-qm', 'diagnostic')
            with patch.object(guide.qualifier, 'ROOT', root), patch.object(guide.qualifier, 'OUT', root / 'output'):
                self.assertEqual(guide.require_same_source(pin), ['docs/ROADMAP.md'])
                source.write_text('changed production source\n');git('add', 'src');git('commit', '-qm', 'drift')
                with self.assertRaisesRegex(RuntimeError, 'production source drift'):
                    guide.require_same_source(pin)


if __name__ == '__main__':
    unittest.main()
