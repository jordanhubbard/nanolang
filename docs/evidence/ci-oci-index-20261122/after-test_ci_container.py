"""I reject changed image identity before executing instrumented providers."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ci_container as container
import ci_bootstrap_checkpoint as checkpoint


class ImmutableImage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'image.tar'
        self.config = json.dumps({'architecture': 'amd64', 'os': 'linux'}).encode()
        self.image = 'sha256:' + hashlib.sha256(self.config).hexdigest()
        self.identity = {'version': 1, 'image_id': self.image, 'config_id': self.image, 'platform': 'linux/amd64',
                         'archive_sha256': 'a' * 64}

    def archive_image(self, config=None, entries=1):
        config = self.config if config is None else config
        with tarfile.open(self.archive, 'w:') as archive:
            for name, value in [('manifest.json', json.dumps([{'Config': 'config.json'}] * entries).encode()),
                                ('config.json', config)]:
                member = tarfile.TarInfo(name); member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
        return container.digest(self.archive)

    def test_actual_archive_config_and_platform_binding(self):
        digest = self.archive_image()
        container.verify_archive(self.archive, digest, self.image, 'linux/amd64')
        for sha, image, platform in [(digest, 'sha256:' + 'b' * 64, 'linux/amd64'),
                                     (digest, self.image, 'linux/arm64'),
                                     ('0' * 64, self.image, 'linux/amd64')]:
            with self.subTest(sha=sha, image=image, platform=platform), self.assertRaises((ValueError, KeyError)):
                container.verify_archive(self.archive, sha, image, platform)
        digest = self.archive_image(entries=2)
        with self.assertRaises(ValueError):
            container.verify_archive(self.archive, digest, self.image, 'linux/amd64')

    def test_oci_manifest_binds_config_and_refuses_changed_edge(self):
        config_hash = 'sha256:' + hashlib.sha256(self.config).hexdigest()
        for broken in (False, True):
            descriptor = json.dumps({'schemaVersion': 2, 'config': {
                'digest': ('sha256:' + 'd' * 64) if broken else config_hash,
                'size': len(self.config)}}).encode()
            image = 'sha256:' + hashlib.sha256(descriptor).hexdigest()
            with tarfile.open(self.archive, 'w:') as archive:
                for name, value in [('manifest.json', json.dumps([{'Config': 'config.json'}]).encode()),
                                    ('config.json', self.config),
                                    ('index.json', json.dumps({'manifests': [{'digest': image}]}).encode()),
                                    ('blobs/sha256/' + image.split(':')[1], descriptor)]:
                    member = tarfile.TarInfo(name); member.size = len(value)
                    archive.addfile(member, io.BytesIO(value))
            if broken:
                with self.assertRaisesRegex(ValueError, 'binding the exact image config'):
                    container.verify_archive(self.archive, container.digest(self.archive), image, 'linux/amd64')
            else:
                self.assertEqual(container.verify_archive(self.archive, container.digest(self.archive), image,
                                                         'linux/amd64'), config_hash)

    def index_archive(self, mutation=None):
        manifest_type = 'application/vnd.oci.image.manifest.v1+json'
        index_type = 'application/vnd.oci.image.index.v1+json'
        config_hash = 'sha256:' + hashlib.sha256(self.config).hexdigest()
        leaf = {'schemaVersion': 2, 'mediaType': manifest_type,
                'config': {'digest': config_hash, 'size': len(self.config)}}
        if mutation == 'config': leaf['config']['digest'] = 'sha256:' + 'd' * 64
        leaf_bytes = json.dumps(leaf).encode()
        leaf_hash = 'sha256:' + hashlib.sha256(leaf_bytes).hexdigest()
        selected = {'mediaType': manifest_type, 'digest': leaf_hash, 'size': len(leaf_bytes),
                    'platform': {'os': 'linux', 'architecture': json.loads(self.config)['architecture']}}
        if mutation == 'platform': selected['platform']['architecture'] = 'arm64'
        if mutation == 'size': selected['size'] += 1
        if mutation == 'attestation': selected['annotations'] = {'vnd.docker.reference.type': 'attestation-manifest'}
        if mutation == 'nested': selected['mediaType'] = index_type
        root = {'schemaVersion': 2, 'mediaType': index_type, 'manifests': [selected, {
            'mediaType': manifest_type, 'digest': 'sha256:' + 'a' * 64, 'size': 123,
            'platform': {'os': 'unknown', 'architecture': 'unknown'},
            'annotations': {'vnd.docker.reference.type': 'attestation-manifest',
                            'vnd.docker.reference.digest': leaf_hash}}]}
        if mutation == 'ambiguous': root['manifests'].append(selected.copy())
        if mutation == 'index-type': root['mediaType'] = 'unknown/index'
        root_bytes = json.dumps(root).encode()
        image = 'sha256:' + hashlib.sha256(root_bytes).hexdigest()
        members = [('manifest.json', json.dumps([{'Config': 'config.json'}]).encode()),
                   ('config.json', self.config),
                   ('index.json', json.dumps({'manifests': [{'digest': image}]}).encode()),
                   ('blobs/sha256/' + image.split(':')[1], root_bytes),
                   ('blobs/sha256/' + leaf_hash.split(':')[1], leaf_bytes)]
        with tarfile.open(self.archive, 'w:') as archive:
            for name, raw in members:
                if name.endswith(leaf_hash.split(':')[1]):
                    if mutation == 'missing': continue
                    if mutation == 'hash': raw = raw[:-1] + b']'
                    if mutation == 'oversize': raw = b'x' * (1024 * 1024 + 1)
                member = tarfile.TarInfo(name); member.size = len(raw)
                if mutation == 'symlink' and name.endswith(leaf_hash.split(':')[1]):
                    member.type = tarfile.SYMTYPE; member.linkname = 'config.json'; member.size = 0
                    archive.addfile(member)
                else:
                    archive.addfile(member, io.BytesIO(raw))
        return image, config_hash

    def test_oci_index_preserves_runnable_config_and_ignores_attestation_execution(self):
        for architecture in ('amd64', 'arm64'):
            with self.subTest(architecture=architecture):
                self.config = json.dumps({'architecture': architecture, 'os': 'linux'}).encode()
                image, config = self.index_archive()
                self.assertEqual(container.verify_archive(self.archive, container.digest(self.archive),
                                                         image, 'linux/' + architecture), config)

    def test_oci_index_refuses_changed_ambiguous_or_nonrunnable_edges(self):
        for mutation in ('config', 'platform', 'size', 'attestation', 'nested', 'ambiguous',
                         'index-type', 'missing', 'hash', 'oversize', 'symlink'):
            with self.subTest(mutation=mutation):
                image, _ = self.index_archive(mutation)
                with self.assertRaises((ValueError, KeyError)):
                    container.verify_archive(self.archive, container.digest(self.archive), image, 'linux/amd64')

    def test_changed_archive_refuses_before_docker_load(self):
        self.archive_image()
        env = {'NANO_CI_IMAGE_ID': self.image, 'NANO_CI_IMAGE_PLATFORM': 'linux/amd64',
               'NANO_CI_ARCHIVE_SHA256': 'b' * 64}
        with mock.patch.dict(os.environ, env), mock.patch.object(container, 'docker') as docker:
            with self.assertRaises(ValueError): container.load(self.archive)
            docker.assert_not_called()

    def test_checkpoint_requires_execution_receipt_not_environment_claim(self):
        path = self.root / 'identity.json'; path.write_text(json.dumps(self.identity))
        marker = self.root / 'dockerenv'; marker.touch()
        with mock.patch.object(checkpoint.platform, 'machine', return_value='x86_64'):
            for outer in ('20261012.334.1', '20261018.339.1'):
                with mock.patch.dict(os.environ, {'ImageVersion': outer}):
                    self.assertEqual(checkpoint.execution_image(path, marker), self.identity)
            marker.unlink()
            with mock.patch.dict(os.environ, {'NANO_CI_IMAGE_ID': self.image}):
                with self.assertRaises(ValueError): checkpoint.execution_image(path, marker)
            marker.touch()
            for key, value in [('version', 2), ('image_id', 'ubuntu:latest'),
                               ('archive_sha256', 'short'), ('platform', 'linux/arm64')]:
                changed = {**self.identity, key: value}; path.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    checkpoint.execution_image(path, marker)

    def test_actual_container_mismatch_refuses_start_and_removes_container(self):
        step = self.root / 'step.sh'; step.write_text('exit 0\n')
        (self.root / 'ci-loaded-image.json').write_text(json.dumps(self.identity))
        env = {'RUNNER_TEMP': str(self.root), 'NANO_CI_IMAGE_ID': self.image,
               'NANO_CI_IMAGE_PLATFORM': 'linux/amd64', 'NANO_CI_ARCHIVE_SHA256': 'a' * 64}
        # The Docker API response is injected only for this rejection boundary.
        def docker(*args, **kwargs):
            if args[0] == 'create': return 'created'
            if args[0] == 'inspect':
                return json.dumps([{'Image': 'sha256:' + 'c' * 64, 'HostConfig': {'ReadonlyRootfs': True}}])
            self.fail(args)
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(container, 'inspect_image', return_value={'Id': self.image}), mock.patch.object(container, 'docker', side_effect=docker), \
             mock.patch.object(container.subprocess, 'run') as run:
            with self.assertRaises(ValueError): container.shell(step)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][:3], ['docker', 'rm', '--force'])
        self.assertFalse(list(self.root.glob('container-identity-*.json')))


if __name__ == '__main__':
    unittest.main()
