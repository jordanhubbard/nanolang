"""I exercise real checkpoint archives without claiming compiler qualification."""
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ci_bootstrap_checkpoint as checkpoint


class CheckpointArchive(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'checkout'; self.root.mkdir()
        self.archive = self.base / 'checkpoint.tar.gz'
        (self.root / 'tool').write_bytes(b'actual provider bytes\n')
        (self.root / 'tool').chmod(0o755)
        os.utime(self.root / 'tool', ns=(1234567890123456789, 1234567890123456789))
        (self.root / 'alias').symlink_to('tool')
        self.metadata = {'root': str(self.root), 'commit': 'exact', 'environment': {'flags': 'same'}}

    def pack(self):
        return checkpoint.write_archive(self.root, ['tool', 'alias'], self.metadata, self.archive)

    def rewrite(self, change):
        with tarfile.open(self.archive, 'r:gz') as source:
            entries = [(member, source.extractfile(member).read() if member.isfile() else None)
                       for member in source.getmembers()]
        entries = change(entries)
        with tarfile.open(self.archive, 'w:gz') as output:
            for member, data in entries:
                if data is not None: member.size = len(data)
                output.addfile(member, io.BytesIO(data) if data is not None else None)

    def test_real_archive_preserves_bytes_modes_links_and_nanosecond_times(self):
        original = self.pack()
        verified = checkpoint.read_archive(self.archive, self.root)
        self.assertEqual(verified, original)
        (self.root / 'tool').write_bytes(b'changed')
        (self.root / 'alias').unlink()
        checkpoint.restore_files(self.archive, self.root, verified)
        self.assertEqual(checkpoint.record(self.root / 'tool'), original['files']['tool'])
        self.assertEqual(checkpoint.record(self.root / 'alias'), original['files']['alias'])
        self.assertEqual((self.root / 'alias').read_bytes(), b'actual provider bytes\n')

    def test_changed_bytes_duplicate_entries_and_unlisted_paths_refuse(self):
        for kind in ('bytes', 'duplicate', 'unlisted', 'hardlink'):
            with self.subTest(kind=kind):
                self.pack()
                def change(entries):
                    if kind == 'duplicate': return entries + [entries[-1]]
                    if kind == 'unlisted':
                        member=tarfile.TarInfo('../outside'); return entries + [(member,b'no')]
                    for member, data in entries:
                        if member.name == 'files/tool':
                            if kind == 'hardlink':
                                member.type=tarfile.LNKTYPE; member.linkname='/tmp/other'
                                return [(m,None if m is member else d) for m,d in entries]
                            return [(m,b'corrupted provider bytes' if m is member else d) for m,d in entries]
                    self.fail('missing fixture entry')
                self.rewrite(change)
                with self.assertRaises(ValueError): checkpoint.read_archive(self.archive,self.root)
                self.assertEqual((self.root/'tool').read_bytes(),b'actual provider bytes\n')

    def test_traversal_and_escaping_links_refuse_before_write(self):
        self.pack()
        def change(entries):
            result=[]
            for member,data in entries:
                if member.name=='checkpoint.json':
                    manifest=json.loads(data); manifest['files']['../escape']=manifest['files'].pop('tool')
                    data=json.dumps(manifest).encode()
                if member.name=='files/tool': member.name='files/../escape'
                result.append((member,data))
            return result
        self.rewrite(change)
        with self.assertRaises(ValueError): checkpoint.read_archive(self.archive,self.root)
        (self.root/'alias').unlink();(self.root/'alias').symlink_to('../outside')
        with self.assertRaises(ValueError): self.pack()
        self.assertFalse((self.base/'outside').exists())

    def test_existing_parent_escape_refuses_all_restore_writes(self):
        (self.root/'nested').mkdir();(self.root/'nested/file').write_text('original')
        manifest=checkpoint.write_archive(self.root,['nested/file'],self.metadata,self.archive)
        (self.root/'nested/file').unlink();(self.root/'nested').rmdir()
        outside=self.base/'outside';outside.mkdir();(outside/'file').write_text('sentinel')
        (self.root/'nested').symlink_to(outside)
        with self.assertRaises(ValueError): checkpoint.restore_files(self.archive,self.root,manifest)
        self.assertEqual((outside/'file').read_text(),'sentinel')

    def test_incomplete_generation_and_changed_identity_refuse(self):
        receipt = {'phase': 'stage1', 'steps': [dict(label=name, exit_code=0) for name in checkpoint.LABELS]}
        checkpoint.require_stage1(receipt)
        for change in ('phase', 'missing', 'unfinished', 'failed'):
            altered = json.loads(json.dumps(receipt))
            if change == 'phase': altered['phase'] = 'building-stage1'
            elif change == 'missing': altered['steps'].pop()
            elif change == 'unfinished': altered['steps'][-1].pop('exit_code')
            else: altered['steps'][-1]['exit_code'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint.require_stage1(altered)
        manifest = {**self.metadata, 'tracked': {'tool': 'recorded'},
                    'files': {'tool': {'sha256': 'recorded'}}}
        checkpoint.require_identity(manifest, self.root, 'exact', {'tool':'recorded'})
        altered = json.loads(json.dumps(manifest)); altered['files']['tool']['sha256'] = 'different archive source'
        with self.assertRaises(ValueError):
            checkpoint.require_identity(altered, self.root, 'exact', {'tool':'recorded'})
        for root, commit, sources in ((self.base,'exact',{'tool':'recorded'}),
                                      (self.root,'different',{'tool':'recorded'}),
                                      (self.root,'exact',{'tool':'changed'})):
            with self.assertRaises(ValueError): checkpoint.require_identity(manifest,root,commit,sources)

    def test_full_checkpoint_requires_every_success_and_raw_equality(self):
        receipt = {'phase': 'stage2', 'raw_stage1_stage2_equal': True,
                   'steps': [dict(label=name, exit_code=0) for name in checkpoint.FINAL_LABELS]}
        checkpoint.require_complete(receipt, 'stage2')
        for index in range(len(receipt['steps'])):
            altered = json.loads(json.dumps(receipt)); altered['steps'][index]['exit_code'] = 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                checkpoint.require_complete(altered, 'stage2')
        for change in ('raw', 'missing', 'repeated'):
            altered = json.loads(json.dumps(receipt))
            if change == 'raw': altered['raw_stage1_stage2_equal'] = False
            elif change == 'missing': altered['steps'].pop()
            else: altered['steps'].append(altered['steps'][-1])
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint.require_complete(altered, 'stage2')

    def test_transferred_complete_checkpoint_continues_in_both_phases(self):
        current = {'phase': 'stage2', 'raw_stage1_stage2_equal': True, 'work': '/exact/work',
                   'steps': [dict(label=name, exit_code=0) for name in checkpoint.FINAL_LABELS],
                   'artifacts': {'first': 'hash-one', 'second': 'hash-two'}}
        for phase in ('stage1', 'stage2'):
            prior = json.loads(json.dumps(current))
            if phase == 'stage1':
                prior['phase'] = phase; prior['steps'] = prior['steps'][:len(checkpoint.LABELS)]
                del prior['artifacts']['second']
            metadata = {**self.metadata, 'receipt': prior, 'phase': phase}
            checkpoint.write_archive(self.root, ['tool', 'alias'], metadata, self.archive)
            manifest = checkpoint.read_archive(self.archive, self.root)
            checkpoint.restore_files(self.archive, self.root, manifest)
            checkpoint.require_continuation(current, manifest['receipt'], phase)
            for change in ('work', 'step', 'artifact'):
                altered = json.loads(json.dumps(current))
                if change == 'work': altered['work'] = '/replacement'
                elif change == 'step': altered['steps'][0]['log_sha256'] = 'replacement'
                else: altered['artifacts']['first'] = 'replacement'
                with self.subTest(phase=phase, change=change), self.assertRaises(ValueError):
                    checkpoint.require_continuation(altered, prior, phase)

    def test_member_beneath_archived_link_refuses(self):
        records={'alias':checkpoint.record(self.root/'alias'),
                 'alias/child':checkpoint.record(self.root/'tool')}
        with self.assertRaises(ValueError): checkpoint.validate_links(records,self.root)


if __name__=='__main__': unittest.main()
