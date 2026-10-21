"""I retain exact hosted selection and unchanged executable arguments."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from scripts import ci_select_capture_clang as selection


class CaptureSelection(unittest.TestCase):
    def fixture(self, root):
        developer = root/'Xcode fixture/Contents/Developer'
        tools = developer/'Toolchains/XcodeDefault.xctoolchain/usr/bin'
        tools.mkdir(parents=True)
        for name in ('clang', 'as'):
            (tools/name).write_text(name)
        sdk = root/'SDK'; sdk.mkdir()
        pins = {name: selection.digest(tools/name) for name in ('clang', 'as')}
        return developer, tools, sdk, pins

    def test_exact_selection_records_tools_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); developer, tools, sdk, pins = self.fixture(root)
            with patch.dict(os.environ, GITHUB_ENV=str(root/'env'), GITHUB_PATH=str(root/'path')):
                with patch.object(selection.subprocess, 'check_output', side_effect=[str(sdk)+'\n', '27.0\n',
                        'Apple clang version 21.0.0 (clang-2100.3.34.2)\n']) as query:
                    record = selection.select(root/'out', developer, pins, sdk)
            self.assertEqual(query.call_count, 3)
            self.assertEqual(record['tools']['clang']['sha256'], pins['clang'])
            self.assertIn('SDKROOT='+str(sdk), (root/'env').read_text())
            self.assertEqual((root/'path').read_text(), str(root/'out/selected-tools')+'\n')
            self.assertEqual(query.call_args.kwargs['env']['DEVELOPER_DIR'], str(developer))
            self.assertFalse((root/'out/selected-tools/cc').is_symlink())

    def test_changed_compiler_or_backend_refuses_before_publication(self):
        for name in ('clang', 'as'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); developer, tools, sdk, pins = self.fixture(root)
                (tools/name).write_text('changed')
                with self.assertRaisesRegex(ValueError, 'identity'):
                    selection.select(root/'out', developer, pins)
                self.assertFalse((root/'out/selected-tools').exists())

    def test_missing_sdk_or_wrong_report_refuses(self):
        for sdk_valid in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); developer, tools, sdk, pins = self.fixture(root)
                responses = [str(sdk if sdk_valid else root/'missing'), '27.0\n', 'unqualified clang\n']
                with patch.object(selection.subprocess, 'check_output', side_effect=responses):
                    with self.assertRaises(ValueError):selection.select(root/'out', developer, pins, sdk)
                self.assertFalse((root/'out/selected-tools').exists())

    def test_different_existing_sdk_or_version_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); developer, tools, sdk, pins = self.fixture(root)
            other = root/'other-sdk'; other.mkdir()
            for responses in ([str(other)], [str(sdk), '28.0\n']):
                with patch.object(selection.subprocess, 'check_output', side_effect=responses):
                    with self.assertRaises(ValueError):selection.select(root/'out', developer, pins, sdk)
                self.assertFalse((root/'out/selected-tools').exists())

    def test_wrapper_preserves_arguments_and_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); compiler = root/'exact compiler'
            compiler.write_text('#!'+selection.sys.executable+
                '\nimport json,sys\nprint(json.dumps(sys.argv))\nsys.exit(7)\n')
            compiler.chmod(0o700)
            wrapper = root/'cc'; wrapper.write_text(selection.wrapper_source(compiler)); wrapper.chmod(0o700)
            args = ['-fno-integrated-as', '-Q', 'a b', '', 'quote"value']
            result = subprocess.run([wrapper, *args], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(json.loads(result.stdout), [str(compiler), *args])
            self.assertEqual(result.stderr, '')

    def test_workflow_preserves_corpora_and_build_compiler(self):
        text = (Path(__file__).resolve().parents[1]/'.github/workflows/retirement-qualification.yml').read_text()
        self.assertIn('os: [ubuntu-latest, xcode-27]', text)
        self.assertIn('corpus: [retirement, compiler-contracts]', text)
        self.assertIn('run: python3 scripts/ci_select_capture_clang.py', text)
        self.assertIn('test_source_snapshots', text)
        self.assertIn('timeout-minutes: 360', text)
        self.assertIn('"$(brew --prefix llvm)"', text)


if __name__ == '__main__':unittest.main()
