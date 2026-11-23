"""I exercise real checkpoint archives without claiming compiler qualification."""
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

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

    def test_compact_source_membership_preserves_identity_and_legacy_reading(self):
        self.metadata['tracked'] = {'tool': checkpoint.record(self.root / 'tool')['sha256']}
        original = self.pack()
        with tarfile.open(self.archive) as archive:
            wire = json.load(archive.extractfile('checkpoint.json'))
        self.assertEqual(wire['version'], 2)
        self.assertEqual(wire['tracked'], [1])
        decoded = checkpoint.read_archive(self.archive, self.root)
        self.assertEqual(decoded, original)
        checkpoint.require_identity(decoded, str(self.root), 'exact', self.metadata['tracked'])
        for names in ([1, 1], [2], [-1], [True], [1, 0], ['tool'], {'tool': 'digest'}):
            changed = {**wire, 'tracked': names}
            with self.subTest(names=names), self.assertRaises(ValueError):
                checkpoint.decode_manifest(json.dumps(changed))
        legacy = {**wire, 'version': 1, 'tracked': self.metadata['tracked']}
        self.assertEqual(checkpoint.decode_manifest(json.dumps(legacy)), original)
        self.metadata['tracked']['tool'] = 'incorrect'
        with self.assertRaises(ValueError): self.pack()

    def test_large_evidence_manifest_fits_without_relaxing_bound(self):
        # I model the 84,263 tracked paths at the failing hosted source pin.
        names = ['docs/evidence/retained-run-' + str(i).zfill(8) +
                 '/compiler-source-and-native-qualification/retained-command.json'
                 for i in range(84263)]
        value = dict(kind='file', sha256='a' * 64, size=1024, mode=420,
                     mtime_ns=1792340000000000000)
        manifest = dict(version=1, files={n: value for n in names},
                        tracked={n: value['sha256'] for n in names})
        old = json.dumps(manifest, sort_keys=True).encode()
        self.assertGreater(len(old), checkpoint.MAX_MANIFEST_BYTES)
        encoded = checkpoint.encode_manifest(manifest)
        self.assertLess(len(encoded), checkpoint.MAX_MANIFEST_BYTES)
        self.assertEqual(checkpoint.decode_manifest(encoded), manifest)
        with self.assertRaises(ValueError):
            checkpoint.encode_manifest({**manifest, 'extra': 'x' * checkpoint.MAX_MANIFEST_BYTES})

    def test_source_directories_survive_fresh_checkout_and_detect_addition(self):
        nested = self.root / 'src' / 'nested'
        nested.mkdir(parents=True)
        (nested / 'input').write_text('source')
        (self.root / 'src' / '.cache').mkdir()
        names = checkpoint.visible_source_directories(self.root)
        self.assertEqual(names, {'src', 'src/nested'})
        for name in names:
            os.utime(self.root / name, ns=(1500000000000000000, 1500000000000000000))
        manifest = checkpoint.write_archive(self.root, sorted(names) + ['src/nested/input'],
                                            self.metadata, self.archive)
        fresh = self.base / 'fresh'; fresh.mkdir()
        verified = checkpoint.read_archive(self.archive, fresh)
        checkpoint.restore_files(self.archive, fresh, verified)
        for name in names:
            self.assertEqual(checkpoint.record(fresh / name), manifest['files'][name])
        (fresh / 'product').write_text('completed')
        os.utime(fresh / 'product', ns=(1600000000000000000, 1600000000000000000))
        (fresh / 'Makefile').write_text('product: src/nested/.\n\t@echo REBUILD\n')
        def plan():
            result = subprocess.run(['make', '-n', 'product'], cwd=fresh,
                                    capture_output=True, text=True, check=True)
            return result.stdout
        self.assertNotIn('REBUILD', plan())
        (fresh / 'src/nested/new-source').write_text('added')
        self.assertIn('REBUILD', plan())
        os.utime(fresh / 'src/nested', ns=(1500000000000000000, 1500000000000000000))
        self.assertNotIn('REBUILD', plan())
        (fresh / 'src/nested/new-source').unlink()
        self.assertIn('REBUILD', plan())

    def test_directory_members_and_conflicts_refuse_before_restore(self):
        (self.root / 'src').mkdir()
        for kind in ('file', 'link'):
            checkpoint.write_archive(self.root, ['src', 'tool'], self.metadata, self.archive)
            def change(entries):
                for member, data in entries:
                    if member.name == 'files/src':
                        member.type = tarfile.REGTYPE if kind == 'file' else tarfile.SYMTYPE
                        member.linkname = 'tool' if kind == 'link' else ''
                return entries
            self.rewrite(change)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                checkpoint.read_archive(self.archive, self.root)
        manifest = checkpoint.write_archive(self.root, ['src', 'tool'], self.metadata, self.archive)
        (self.root / 'src').rmdir(); (self.root / 'src').write_text('sentinel')
        (self.root / 'tool').write_text('unchanged until validation')
        with self.assertRaises(ValueError):
            checkpoint.restore_files(self.archive, self.root, manifest)
        self.assertEqual((self.root / 'src').read_text(), 'sentinel')
        self.assertEqual((self.root / 'tool').read_text(), 'unchanged until validation')
        records = {'tool': checkpoint.record(self.root / 'tool'),
                   'tool/child': checkpoint.record(self.root / 'tool')}
        with self.assertRaises(ValueError): checkpoint.validate_links(records, self.root)
        (self.root / 'src').unlink(); (self.root / 'src').symlink_to(self.base)
        with self.assertRaises(ValueError): checkpoint.visible_source_directories(self.root)
        with self.assertRaises(ValueError):
            checkpoint.restore_files(self.archive, self.root, manifest)

    def test_actual_schema_header_rule_has_no_phantom_rebuild(self):
        source = (Path(__file__).resolve().parents[1] / 'Makefile.gnu').read_text()
        start = source.index('$(SRC_DIR)/generated/compiler_schema.h: $(SCHEMA_STAMP)')
        rule = source[start:source.index('\n\n', start)]
        generated = self.root / 'src/generated'; generated.mkdir(parents=True)
        header = generated / 'compiler_schema.h'
        for name, stamp in ((header, 1500000000), (self.root / 'schema.stamp', 1500000001),
                            (self.root / 'product', 1500000002)):
            name.write_text('retained'); os.utime(name, (stamp, stamp))
        makefile = 'SRC_DIR=src\nSCHEMA_STAMP=schema.stamp\n' + rule + \
            '\nproduct: src/generated/compiler_schema.h\n\t@echo REBUILD\n'
        (self.root / 'Makefile').write_text(makefile)
        def plan():
            return subprocess.run(['make', '-n', 'product'], cwd=self.root,
                                  capture_output=True, text=True, check=True).stdout
        self.assertNotIn('REBUILD', plan())
        os.utime(header, (1500000003, 1500000003))
        self.assertIn('REBUILD', plan())
        # I exercise the existing stamp producer's side effect, not a new
        # direct header recipe: a fresh schema regenerates a missing header.
        header.unlink()
        (self.root / 'schema.input').write_text('new schema')
        os.utime(self.root / 'schema.input', (1500000004, 1500000004))
        with (self.root / 'Makefile').open('a') as output:
            output.write('schema.stamp: schema.input\n\t@echo generated > src/generated/compiler_schema.h\n\t@touch schema.stamp\n')
        result = subprocess.run(['make', 'product'], cwd=self.root,
                                capture_output=True, text=True, check=True)
        self.assertEqual(header.read_text(), 'generated\n')
        self.assertIn('REBUILD', result.stdout)

    def test_member_beneath_archived_link_refuses(self):
        records={'alias':checkpoint.record(self.root/'alias'),
                 'alias/child':checkpoint.record(self.root/'tool')}
        with self.assertRaises(ValueError): checkpoint.validate_links(records,self.root)




class CompactCheckpointTables(unittest.TestCase):
    def test_current_failure_count_round_trips_every_identity_under_original_bound(self):
        # I retain the failing source count and its160840 distinct content groups.
        count, unique = 346663, 160840
        hashes = [format(index, '064x') for index in range(unique)]
        files = {}
        for index in range(count):
            name = ('docs/evidence/retained-' + str(index // 100).zfill(8) +
                    '/compiler-source-and-native-qualification/source-' +
                    str(index % 263248).zfill(12) + '.stdout')
            files[name] = dict(kind='file', sha256=hashes[index % unique],
                               size=(1 << 63) - 1, mode=0o644,
                               mtime_ns=1792340000000000000 + index * 1000000)
        manifest = dict(version=1, files=files, tracked={name: value['sha256'] for name, value in files.items()})
        encoded = checkpoint.encode_manifest(manifest)
        self.assertEqual(json.loads(encoded)['version'], 3)
        self.assertLessEqual(len(encoded), 32 * 1024 * 1024)
        self.assertEqual(checkpoint.MAX_MANIFEST_BYTES, 32 * 1024 * 1024)
        decoded = checkpoint.decode_manifest(encoded)
        self.assertEqual(decoded, manifest)
        names = sorted(files)
        decoded['files'][names[0]]['mtime_ns'] += 1
        self.assertEqual(decoded['files'][names[-1]], files[names[-1]])

    def test_compact_tables_reject_aliases_bad_indices_and_lost_identity_fields(self):
        files = {
            'group': dict(kind='directory', mode=0o755, mtime_ns=-123),
            'group/alias': dict(kind='link', mode=0o777, mtime_ns=1234567890123456789,
                                target='tool', sha256=checkpoint.sha(b'tool')),
            'group/tool': dict(kind='file', mode=0o751, mtime_ns=1234567890123456791,
                               size=19, sha256='1' * 64),
            'other/tool': dict(kind='file', mode=0o751, mtime_ns=1234567890123456799,
                               size=19, sha256='1' * 64),
        }
        logical = {'version': 1, 'files': files, 'tracked': {'group/tool': '1' * 64}}
        compact = checkpoint.compact_file_records(files)
        wire = {'version': 3, 'files': compact, 'tracked': [2]}
        self.assertEqual(checkpoint.decode_manifest(json.dumps(wire)), logical)
        mutations = []
        def add(label, change):
            altered = json.loads(json.dumps(wire))
            change(altered['files'])
            mutations.append((label, altered))
        add('duplicate-directory', lambda c: c['directories'].append(c['directories'][-1]))
        add('parent-traversal', lambda c: c['directories'][0].__setitem__(0, '../escape'))
        add('base-time-type', lambda c: c['directories'][0].__setitem__(1, True))
        add('basename-traversal', lambda c: c['names'].__setitem__(0, '../escape'))
        add('duplicate-name', lambda c: c['names'].append(c['names'][-1]))
        add('unused-name', lambda c: c['names'].append('zzzz-unused'))
        add('unknown-table', lambda c: c.__setitem__('omitted-fields', []))
        add('negative-index', lambda c: c['entries'][0].__setitem__(0, -1))
        add('boolean-index', lambda c: c['entries'][0].__setitem__(1, True))
        add('large-index', lambda c: c['entries'][0].__setitem__(2, len(c['identities'])))
        add('negative-time-delta', lambda c: c['entries'][0].__setitem__(3, -1))
        add('string-time-delta', lambda c: c['entries'][0].__setitem__(3, '1'))
        add('duplicate-path', lambda c: c['entries'].append(c['entries'][-1]))
        add('unsorted-paths', lambda c: c['entries'].reverse())
        add('extra-row-field', lambda c: c['entries'][0].append(1))
        add('unknown-kind', lambda c: c['identities'][0].__setitem__(0, 3))
        add('boolean-mode', lambda c: c['identities'][0].__setitem__(1, True))
        add('bad-mode', lambda c: c['identities'][0].__setitem__(1, 0o10000))
        add('bad-file-size', lambda c: c['identities'][0].__setitem__(2, -1))
        add('bad-hash-size', lambda c: c['identities'][0].__setitem__(3, 'x'))
        add('bad-hash-bytes', lambda c: c['identities'][0].__setitem__(3, '~' * 40))
        add('lost-identity-field', lambda c: c['identities'][0].pop())
        add('unlisted-identity-field', lambda c: c['identities'][0].append(0))
        tracked_directory = json.loads(json.dumps(wire)); tracked_directory['tracked'] = [0]
        mutations.append(('tracked-directory', tracked_directory))
        for label, altered in mutations:
            with self.subTest(label=label), self.assertRaises(ValueError):
                checkpoint.decode_manifest(json.dumps(altered))

    def test_version3_real_archive_restores_links_directories_modes_and_times(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); root = base/'checkout'; root.mkdir()
            names = []
            for index in range(8):
                parent = root/('group-'+str(index)); parent.mkdir()
                tool = parent/'tool'; tool.write_bytes(b'actual repeated provider bytes\n'); tool.chmod(0o751)
                os.utime(tool, ns=(1234567890123456700+index, 1234567890123456700+index))
                os.utime(parent, ns=(1234567890123456800+index, 1234567890123456800+index))
                names.extend((str(parent.relative_to(root)), str(tool.relative_to(root))))
            (root/'alias').symlink_to('group-0/tool'); names.append('alias')
            archive = base/'checkpoint.tar.gz'
            metadata = dict(root=str(root), commit='same', tracked={name:checkpoint.record(root/name)['sha256']
                for name in names if (root/name).is_file()})
            original = checkpoint.write_archive(root, names, metadata, archive)
            with tarfile.open(archive) as stored: old_wire = stored.extractfile('checkpoint.json').read()
            self.assertEqual(json.loads(old_wire)['version'], 2)
            # I lower the fixture bound to select the real compact encoder; production stays32MiB.
            with patch.object(checkpoint, 'MAX_MANIFEST_BYTES', len(old_wire)-1):
                compact = checkpoint.write_archive(root, names, metadata, archive)
                with tarfile.open(archive) as stored: wire = stored.extractfile('checkpoint.json').read()
                self.assertEqual(json.loads(wire)['version'], 3)
                self.assertLess(len(wire), len(old_wire))
                self.assertEqual(compact, original)
                decoded = checkpoint.read_archive(archive, root)
                self.assertEqual(decoded, original)
                (root/'group-0/tool').write_bytes(b'changed provider')
                (root/'alias').unlink()
                checkpoint.restore_files(archive, root, decoded)
                for name in names:
                    self.assertEqual(checkpoint.record(root/name), original['files'][name])
            with tarfile.open(archive) as source:
                members = [(member, source.extractfile(member).read() if member.isfile() else None)
                           for member in source.getmembers()]
            with tarfile.open(archive, 'w:gz') as changed:
                for member, payload in members:
                    if member.name == 'checkpoint.json':
                        value = json.loads(payload)
                        link = next(row for row in value['files']['identities'] if row[0] == 1)
                        link[2] = '../outside'
                        payload = json.dumps(value).encode(); member.size = len(payload)
                    changed.addfile(member, io.BytesIO(payload) if payload is not None else None)
            with self.assertRaises(ValueError): checkpoint.read_archive(archive, root)
            self.assertFalse((base/'outside').exists())
            self.assertEqual((root/'group-0/tool').read_bytes(), b'actual repeated provider bytes\n')


class CheckpointLinkGraph(unittest.TestCase):
    def raw_archive(self, root, names, archive, version):
        # I construct reader inputs independently of the writer's link guard.
        records = {name: checkpoint.record(root / name) for name in sorted(names)}
        manifest = dict(version=1, root=str(root), commit='link-fixture', files=records,
                        tracked={name: value['sha256'] for name, value in records.items()
                                 if value['kind'] == 'file'})
        if version == 1:
            encoded = json.dumps(manifest).encode()
        elif version == 2:
            encoded = checkpoint.encode_manifest(manifest)
            self.assertEqual(json.loads(encoded)['version'], 2)
        else:
            ordered = sorted(records)
            wire = {**manifest, 'version': 3, 'files': checkpoint.compact_file_records(records),
                    'tracked': [index for index, name in enumerate(ordered)
                                if name in manifest['tracked']]}
            encoded = json.dumps(wire).encode()
        self.assertLessEqual(len(encoded), 32 * 1024 * 1024)
        with tarfile.open(archive, 'w:gz', format=tarfile.PAX_FORMAT) as output:
            header = tarfile.TarInfo('checkpoint.json'); header.size = len(encoded)
            output.addfile(header, io.BytesIO(encoded))
            for name, value in records.items():
                member = tarfile.TarInfo('files/' + name); member.mode = value['mode']
                if value['kind'] == 'directory':
                    member.type = tarfile.DIRTYPE; output.addfile(member)
                elif value['kind'] == 'link':
                    member.type = tarfile.SYMTYPE; member.linkname = value['target']
                    output.addfile(member)
                else:
                    data = (root / name).read_bytes(); member.size = len(data)
                    output.addfile(member, io.BytesIO(data))
        return manifest

    def test_composed_archived_escape_refuses_pack_and_every_reader_version(self):
        for version in (1, 2, 3):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                base = Path(directory).resolve(); root = base / 'checkout'; root.mkdir()
                (base / 'outside').write_text('outside sentinel')
                (root / 'pivot').symlink_to('.')
                (root / 'escape').symlink_to('pivot/../outside')
                archive = base / 'checkpoint.tar.gz'
                with self.assertRaises(ValueError):
                    checkpoint.write_archive(root, ['pivot', 'escape'], {}, archive)
                self.raw_archive(root, ['pivot', 'escape'], archive, version)
                with self.assertRaises(ValueError): checkpoint.read_archive(archive, root)
                self.assertEqual((base / 'outside').read_text(), 'outside sentinel')

    def test_existing_unarchived_link_escape_refuses_every_reader_version(self):
        for version in (1, 2, 3):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                base = Path(directory).resolve(); root = base / 'checkout'; root.mkdir()
                outside = base / 'outside'; outside.mkdir()
                (outside / 'sentinel').write_text('outside sentinel')
                (root / 'pivot').symlink_to(outside)
                (root / 'alias').symlink_to('pivot/sentinel')
                archive = base / 'checkpoint.tar.gz'
                self.raw_archive(root, ['alias'], archive, version)
                with self.assertRaises(ValueError): checkpoint.read_archive(archive, root)
                with self.assertRaises(ValueError):
                    checkpoint.write_archive(root, ['alias'], {}, archive)
                self.assertEqual((outside / 'sentinel').read_text(), 'outside sentinel')

    def test_destination_link_change_refuses_before_any_restore_write(self):
        for version in (1, 2, 3):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                base = Path(directory).resolve(); root = base / 'checkout'; root.mkdir()
                (root / 'pivot').mkdir()
                (root / 'pivot/sentinel').write_text('safe source')
                (root / 'a-tool').write_text('archived provider')
                (root / 'z-alias').symlink_to('pivot/sentinel')
                archive = base / 'checkpoint.tar.gz'
                self.raw_archive(root, ['a-tool', 'z-alias'], archive, version)
                manifest = checkpoint.read_archive(archive, root)
                (root / 'a-tool').write_text('unchanged until validation')
                (root / 'pivot/sentinel').unlink(); (root / 'pivot').rmdir()
                outside = base / 'outside'; outside.mkdir()
                (outside / 'sentinel').write_text('outside sentinel')
                (root / 'pivot').symlink_to(outside)
                with self.assertRaises(ValueError): checkpoint.restore_files(archive, root, manifest)
                self.assertEqual((root / 'a-tool').read_text(), 'unchanged until validation')
                self.assertEqual((outside / 'sentinel').read_text(), 'outside sentinel')

    def test_safe_chains_parent_components_and_absolute_internal_links_restore(self):
        for version in (1, 2, 3):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                base = Path(directory).resolve(); root = base / 'checkout'; root.mkdir()
                (root / 'group').mkdir()
                (root / 'tool').write_text('archived provider')
                (root / 'group/hop').symlink_to('../tool')
                (root / 'alias').symlink_to('group/hop')
                (root / 'absolute').symlink_to(root / 'group/hop')
                (root / 'parent-chain').symlink_to('group/../alias')
                names = ['group', 'tool', 'group/hop', 'alias', 'absolute', 'parent-chain']
                archive = base / 'checkpoint.tar.gz'
                original = self.raw_archive(root, names, archive, version)
                decoded = checkpoint.read_archive(archive, root)
                self.assertEqual(decoded, original)
                for name in ('alias', 'absolute', 'parent-chain', 'group/hop'):
                    (root / name).unlink()
                (root / 'tool').write_text('changed provider')
                checkpoint.restore_files(archive, root, decoded)
                for name in ('alias', 'absolute', 'parent-chain', 'group/hop'):
                    self.assertEqual((root / name).read_text(), 'archived provider')
                for name, value in original['files'].items():
                    self.assertEqual(checkpoint.record(root / name), value)
                checkpoint.write_archive(root, names, {}, base / 'safe-writer.tar.gz')

    def test_cycles_and_overlong_chains_refuse_but_forty_hops_remain_valid(self):
        for version in (1, 2, 3):
            for count in (2, 40, 41):
                with self.subTest(version=version, count=count), tempfile.TemporaryDirectory() as directory:
                    base = Path(directory).resolve(); root = base / 'checkout'; root.mkdir()
                    (root / 'tool').write_text('provider')
                    for index in range(count):
                        target = 'link-' + str(index + 1) if index + 1 < count else ('link-0' if count == 2 else 'tool')
                        (root / ('link-' + str(index))).symlink_to(target)
                    names = ['link-' + str(index) for index in range(count)] + ['tool']
                    archive = base / 'checkpoint.tar.gz'
                    self.raw_archive(root, names, archive, version)
                    if count == 40:
                        checkpoint.read_archive(archive, root)
                        checkpoint.write_archive(root, names, {}, base / 'safe-chain.tar.gz')
                    else:
                        with self.assertRaises(ValueError): checkpoint.read_archive(archive, root)
                        with self.assertRaises(ValueError):
                            checkpoint.write_archive(root, names, {}, base / 'unsafe-chain.tar.gz')


if __name__=='__main__': unittest.main()
