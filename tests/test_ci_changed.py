"""I check that the CI change classifier skips only pure documentation pushes."""
import subprocess
import os
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import ci_changed  # noqa: E402


class CiChanged(unittest.TestCase):
    def test_documentation_only_is_not_code(self):
        for paths in (['docs/ROADMAP.md'], ['docs/evidence/x/README.md', 'CHANGELOG.md'], ['README.md'],
                      ['docs/RELEASE_5.1_STATUS.md', 'docs/NANOISA_ONLY.md', 'CHANGELOG.md']):
            with self.subTest(paths=paths):
                self.assertFalse(ci_changed.classify(paths))

    def test_any_code_or_build_file_is_code(self):
        for paths in (['src/typechecker.c'], ['docs/ROADMAP.md', 'src/typechecker.c'], ['Makefile.gnu'],
                      ['.github/workflows/ci.yml'], ['tests/test_x.py'], ['scripts/fast_gate.py'],
                      ['docs/ROADMAP.md', 'Makefile.gnu'], ['src_nano/nanoc_v06.nano'], ['docs/tool.py']):
            with self.subTest(paths=paths):
                self.assertTrue(ci_changed.classify(paths))

    def test_unknown_or_empty_input_runs_the_long_jobs(self):
        for paths in ([], [''], ['   ']):
            with self.subTest(paths=paths):
                self.assertTrue(ci_changed.classify(paths))

    def test_command_output(self):
        for text, expected in (('docs/a.md\n', 'code=false'), ('docs/a.md\nsrc/a.c\n', 'code=true'), ('', 'code=true')):
            with self.subTest(text=text):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/ci_changed.py')], input=text,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.stdout.strip(), expected)


class GitHubEventClassification(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nano-ci-event-')
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / 'source'
        self.source.mkdir()
        self.git(self.source, 'init', '--quiet')
        self.git(self.source, 'config', 'user.name', 'NanoLang fixture')
        self.git(self.source, 'config', 'user.email', 'fixture@example.invalid')
        self.base = self.commit('README.md', 'I retain my baseline.\n')
        self.code = self.commit('src/feature.c', 'int feature(void) { return 1; }\n')
        self.prose = self.commit('docs/progress.md', 'I record progress.\n')
        self.checkout = Path(self.temporary.name) / 'checkout'
        self.git(Path(self.temporary.name), 'clone', '--quiet', str(self.source), str(self.checkout))

    def git(self, cwd, *args):
        return subprocess.check_output(['git', '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
                                        *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL,
                                       timeout=30).strip()

    def commit(self, name, text):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.git(self.source, 'add', name)
        self.git(self.source, 'commit', '--quiet', '-m', 'I record a fixture input.')
        return self.git(self.source, 'rev-parse', 'HEAD')

    def test_actual_push_prose_range_is_documentation(self):
        self.assertFalse(ci_changed.github_event_changes('push',
                         {'before': self.code, 'after': self.prose}, self.checkout))

    def test_complete_pr_range_keeps_earlier_code_after_a_prose_push(self):
        event = {'before': self.code, 'after': self.prose,
                 'pull_request': {'base': {'sha': self.base}, 'head': {'sha': self.prose}}}
        self.assertFalse(ci_changed.github_event_changes('push', event, self.checkout))
        self.assertTrue(ci_changed.github_event_changes('pull_request', event, self.checkout))

    def test_pure_prose_pr_range_is_documentation(self):
        event = {'pull_request': {'base': {'sha': self.code}, 'head': {'sha': self.prose}}}
        self.assertFalse(ci_changed.github_event_changes('pull_request', event, self.checkout))

    def test_missing_remote_and_unknown_commit_keep_native_jobs(self):
        self.assertTrue(ci_changed.github_event_changes('push',
                        {'before': 'f' * 40, 'after': self.prose}, self.checkout))
        self.git(self.checkout, 'remote', 'remove', 'origin')
        self.assertTrue(ci_changed.github_event_changes('push',
                        {'before': self.code, 'after': self.prose}, self.checkout))

    def test_manual_empty_and_unknown_events_keep_native_jobs(self):
        for name, event in [('workflow_dispatch', {}), ('push', {}), ('pull_request', {}),
                            ('push', {'before': '0' * 40, 'after': self.prose}),
                            ('push', {'before': self.prose, 'after': self.prose})]:
            with self.subTest(name=name, event=event):
                self.assertTrue(ci_changed.github_event_changes(name, event, self.checkout))

    def test_real_cli_reads_the_event_and_fails_closed_for_bad_json(self):
        event = Path(self.temporary.name) / 'event.json'
        event.write_text(json.dumps({'pull_request': {'base': {'sha': self.base},
                                                      'head': {'sha': self.prose}}}))
        env = dict(os.environ, GITHUB_EVENT_PATH=str(event), GITHUB_EVENT_NAME='pull_request')
        for content, expected in [(event.read_text(), 'code=true\n'), ('not json', 'code=true\n')]:
            event.write_text(content)
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/ci_changed.py'), '--github-event'],
                                    cwd=self.checkout, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)


if __name__ == '__main__':
    unittest.main()
