#!/usr/bin/env python3
"""I retain original failed assembler fixtures and diagnose exact link-input refusals."""
import argparse
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from functools import partial
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_source_snapshots import SourceSnapshots


class PreserveFailedDirectory(tempfile.TemporaryDirectory):
    def __init__(self, *args, journal, **kwargs):
        if 'delete' not in inspect.signature(tempfile_directory).parameters:
            raise RuntimeError('I require public TemporaryDirectory delete controls for exact failure retention.')
        kwargs['delete'] = False
        super().__init__(*args, **kwargs)
        self.journal = Path(journal)

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.cleanup()
        else:
            with self.journal.open('a') as log:
                log.write(json.dumps({'root': self.name, 'exception': kind.__name__}) + '\n')
        return False


tempfile_directory = tempfile.TemporaryDirectory


class FocusedLinkInputs(SourceSnapshots):
    def original_cases(self, style, suffix=None, shared_unit=False):
        cases = [(style, 'common', external, shared)
                 for external in (False, True) for shared in (False, True)]
        cases.append((style, 'platform', False, False))
        self.assembler_search_order_phases_and_recovery((style,), suffix=suffix,
            shared_unit=shared_unit, cases=cases)

    def test_xassembler(self):
        self.original_cases('xassembler')

    def test_split_xassembler(self):
        self.original_cases('xassembler-split')

    def test_raw_assembler(self):
        self.original_cases('xassembler-split', suffix='.s')

    def test_preprocessed_assembler(self):
        self.original_cases('xassembler-split', suffix='.S')

    def test_shared_raw_assembler(self):
        self.original_cases('xassembler-split', suffix='.s', shared_unit=True)

    def test_shared_preprocessed_assembler(self):
        self.original_cases('xassembler-split', suffix='.S', shared_unit=True)


def inspect_records(roots):
    records = []
    for root in roots:
        for directory, children, files in os.walk(root, followlinks=False):
            children[:] = [name for name in children if not (Path(directory)/name).is_symlink()]
            for name in files:
                if name != '.link-dependencies':
                    continue
                path = Path(directory)/name
                if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
                    continue
                if path.stat().st_size > 16 * 1024 * 1024:
                    records.append({'path': str(path), 'oversized': True})
                    continue
                data = path.read_bytes()
                offset, entries = 0, []
                while offset < len(data) and len(entries) < 65536:
                    tag = data[offset]
                    end = data.find(b'\0', offset+1)
                    if end < 0:
                        entries.append({'tag': tag, 'unterminated': True})
                        break
                    text = os.fsdecode(data[offset+1:end])
                    item = {'tag': tag, 'text': text}
                    if tag in (16, 17, 64):
                        try:
                            item.update(exists=Path(text).exists(), regular=Path(text).is_file())
                        except OSError as error:
                            item['stat_errno'] = error.errno
                    entries.append(item)
                    offset = end+1
                records.append({'path': str(path), 'bytes': len(data), 'entries': entries})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.test_output/darwin-link-diagnostic'))
    args = parser.parse_args()
    if sys.platform != 'darwin':
        raise SystemExit('I require actual Darwin execution for this diagnostic.')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    journal = output/'retained-roots.jsonl'
    if journal.exists():
        raise SystemExit('I refuse to overwrite an earlier diagnostic journal.')
    methods = ['test_xassembler', 'test_split_xassembler', 'test_raw_assembler',
               'test_preprocessed_assembler', 'test_shared_raw_assembler',
               'test_shared_preprocessed_assembler']
    suite = unittest.TestSuite(FocusedLinkInputs(name) for name in methods)
    with patch('tempfile.TemporaryDirectory', partial(PreserveFailedDirectory, journal=journal)):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    roots = [json.loads(line)['root'] for line in journal.read_text().splitlines()] if journal.exists() else []
    (output/'link-records.json').write_text(json.dumps(inspect_records(roots), indent=2)+'\n')
    summary = {'methods': methods, 'selected_configurations': 30,
        'targeted_historical_failures': 24, 'neighbor_controls': 6,
        'methods_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped), 'roots': roots,
        'full_source_snapshot_corpus': False, 'native_lifecycle_admitted': False}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    raise SystemExit(main())
