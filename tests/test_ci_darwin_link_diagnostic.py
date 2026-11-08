"""I preserve first-failure bytes and ordinary successful cleanup through public APIs."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from scripts.ci_darwin_link_diagnostic import PreserveFailedDirectory, inspect_records


class DarwinLinkRetention(unittest.TestCase):
    def test_direct_script_entry_point(self):
        script = Path(__file__).resolve().parents[1]/'scripts/ci_darwin_link_diagnostic.py'
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp,
                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--output', result.stdout)

    def test_success_removes_only_its_owned_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); journal=root/'journal.jsonl'
            outside=root/'retained';outside.write_bytes(b'I remain.')
            with PreserveFailedDirectory(dir=root,journal=journal) as selected:
                (Path(selected)/'success').write_bytes(b'success')
            self.assertFalse(Path(selected).exists())
            self.assertFalse(journal.exists())
            self.assertEqual(outside.read_bytes(),b'I remain.')

    def test_failure_retains_exact_files_links_and_raw_link_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);journal=root/'journal.jsonl'
            with self.assertRaises(AssertionError):
                with PreserveFailedDirectory(dir=root,journal=journal) as selected:
                    work=Path(selected)
                    content=work/'native';content.write_bytes(b'\x00I retain original bytes.\xff')
                    content.chmod(0o751)
                    (work/'alias').symlink_to('native')
                    record=b'\x00@(#)PROGRAM:ld PROJECT:ld-test\0\x10'+bytes(content)+b'\0'
                    (work/'.link-dependencies').write_bytes(record)
                    raise AssertionError('I retain the first failure.')
            row=json.loads(journal.read_text())
            self.assertEqual(row['root'],selected)
            self.assertEqual(row['exception'],'AssertionError')
            self.assertEqual(content.read_bytes(),b'\x00I retain original bytes.\xff')
            self.assertEqual(content.stat().st_mode & 0o777,0o751)
            self.assertTrue((work/'alias').is_symlink())
            self.assertEqual((work/'.link-dependencies').read_bytes(),record)
            parsed=inspect_records([selected])
            self.assertEqual(parsed[0]['entries'][1]['tag'],16)
            self.assertTrue(parsed[0]['entries'][1]['regular'])
            self.assertEqual(parsed[0]['entries'][1]['text'],str(content))
