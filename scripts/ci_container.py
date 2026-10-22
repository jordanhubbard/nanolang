#!/usr/bin/env python3
"""I transport and execute one verified instrumented CI image, without a registry push."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = '/etc/nanolang-ci-container.json'
PASS_ENV = ('CFLAGS', 'LDFLAGS', 'CC', 'NANO_CC', 'NANO_CFLAGS', 'NANO_LDFLAGS',
            'BOOTSTRAP_NANOISA_TIMEOUT', 'NANO_SHADOW_TIMEOUT_SECONDS', 'NANO_SHADOW_TRACE',
            'NANO_VERIFY_CORPUS_LOG_ROOT', 'CMD_TIMEOUT', 'ASAN_OPTIONS', 'UBSAN_OPTIONS',
            'NANOLANG_COMPILER', 'NANO_VM_EXAMPLE_SHADOW_TIMEOUT_SECONDS')


def require(condition, message):
    if not condition:
        raise ValueError('I require ' + message)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def docker(*args, capture=True, timeout=120):
    result = subprocess.run(['docker', *map(str, args)], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None, timeout=timeout)
    return result.stdout.strip() if capture else None


def inspect_image(image, platform, config=None):
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', image), 'a content-addressed image ID')
    require(platform in ('linux/amd64', 'linux/arm64'), 'an explicit supported Linux image platform')
    try:
        info = json.loads(docker('image', 'inspect', image))[0]
    except subprocess.CalledProcessError:
        require(config is not None and config != image, 'the transported image in the local image store')
        info = json.loads(docker('image', 'inspect', config))[0]
    require(info['Id'] in (image, config) and info['Os'] + '/' + info['Architecture'] == platform,
            'the exact loaded image config and platform')
    return info


def verify_archive(archive, expected_sha, image, platform):
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha), 'a complete image archive SHA256')
    require(digest(archive) == expected_sha, 'unchanged image archive bytes before loading')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', image), 'a content-addressed image ID')
    # I read, but never extract, the Docker save manifest/config before loading.
    with tarfile.open(archive, 'r:') as source:
        member = source.getmember('manifest.json')
        require(member.isfile() and member.size <= 1024 * 1024, 'a bounded image manifest')
        manifest = json.load(source.extractfile(member))
        require(len(manifest) == 1, 'one transported image')
        config = source.getmember(manifest[0]['Config'])
        require(config.isfile() and config.size <= 1024 * 1024, 'a bounded image config')
        data = source.extractfile(config).read()
        config_digest = 'sha256:' + hashlib.sha256(data).hexdigest()
        if config_digest != image:
            # Docker's containerd store addresses the OCI manifest rather than
            # the config. I verify that extra hash edge, not an asserted alias.
            index_member = source.getmember('index.json')
            require(index_member.isfile() and index_member.size <= 1024 * 1024, 'a bounded OCI image index')
            index = json.load(source.extractfile(index_member))
            require(len(index['manifests']) == 1 and index['manifests'][0]['digest'] == image,
                    'the recorded single OCI image manifest')
            manifest_member = source.getmember('blobs/sha256/' + image.split(':')[1])
            require(manifest_member.isfile() and manifest_member.size <= 1024 * 1024, 'a bounded OCI manifest')
            manifest_bytes = source.extractfile(manifest_member).read()
            require('sha256:' + hashlib.sha256(manifest_bytes).hexdigest() == image,
                    'unchanged OCI manifest bytes')
            descriptor = json.loads(manifest_bytes)
            require(descriptor['config']['digest'] == config_digest and
                    descriptor['config']['size'] == len(data), 'the OCI manifest binding the exact image config')
        info = json.loads(data)
        require(info['os'] + '/' + info['architecture'] == platform, 'the recorded archive platform')
        return config_digest


def outputs(values):
    path = os.environ.get('GITHUB_OUTPUT')
    if path:
        with open(path, 'a') as target:
            for key, value in values.items():
                target.write(f'{key}={value}\n')
    print(json.dumps(values, sort_keys=True))


def build(archive, platform):
    require(platform in ('linux/amd64', 'linux/arm64'), 'an explicit build platform')
    archive.parent.mkdir(parents=True, exist_ok=True)
    image_file = archive.with_suffix('.id')
    docker('build', '--platform', platform, '--iidfile', image_file,
           ROOT / '.github/containers/instrumented', capture=False, timeout=1500)
    image = image_file.read_text().strip()
    inspect_image(image, platform)
    docker('save', '--output', archive, image, capture=False, timeout=300)
    archive_sha = digest(archive)
    config = verify_archive(archive, archive_sha, image, platform)
    outputs({'image-id': image, 'config-id': config, 'archive-sha256': archive_sha, 'platform': platform})


def expected():
    return (os.environ['NANO_CI_IMAGE_ID'], os.environ['NANO_CI_IMAGE_PLATFORM'],
            os.environ['NANO_CI_ARCHIVE_SHA256'])


def load(archive):
    image, platform, archive_sha = expected()
    config = verify_archive(archive, archive_sha, image, platform)
    docker('load', '--input', archive, capture=False, timeout=300)
    inspect_image(image, platform, config)
    proof = Path(os.environ['RUNNER_TEMP']) / 'ci-loaded-image.json'
    proof.write_text(json.dumps({'version': 1, 'image_id': image, 'config_id': config,
                                'platform': platform, 'archive_sha256': archive_sha}, sort_keys=True) + '\n')
    print('I loaded the exact transported instrumented image.')


def shell(script):
    image, platform, archive_sha = expected()
    temporary = Path(os.environ['RUNNER_TEMP']).resolve()
    identity = json.loads((temporary / 'ci-loaded-image.json').read_text())
    require(identity.get('version') == 1 and identity.get('image_id') == image and
            identity.get('platform') == platform and identity.get('archive_sha256') == archive_sha,
            'the previously verified archive load receipt')
    config = identity['config_id']
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', config), 'a verified config digest')
    loaded = inspect_image(image, platform, config)
    execution_image = loaded['Id']
    script = script.resolve()
    require(script.is_file() and script.is_relative_to(temporary), 'a runner-owned step script')
    temporary.mkdir(parents=True, exist_ok=True)
    (temporary / 'container-home').mkdir(exist_ok=True)
    require(re.fullmatch(r'[0-9a-f]{64}', archive_sha), 'a complete transferred archive digest')
    receipt = temporary / ('container-identity-' + uuid.uuid4().hex + '.json')
    receipt.write_text(json.dumps(identity, sort_keys=True) + '\n')
    receipt.chmod(0o444)
    container = 'nanolang-ci-' + uuid.uuid4().hex
    scratch = temporary / (container + '-tmp')
    scratch.mkdir()
    args = ['create', '--name', container, '--platform', platform, '--read-only', '--init',
            '--user', f'{os.getuid()}:{os.getgid()}', '--workdir', '/workspace',
            '--mount', f'type=bind,src={scratch},dst=/tmp', '--tmpfs', '/run',
            '--mount', f'type=bind,src={ROOT},dst=/workspace',
            '--mount', f'type=bind,src={temporary},dst=/ci-temp',
            '--mount', f'type=bind,src={receipt},dst={IDENTITY},readonly',
            '--env', 'HOME=/ci-temp/container-home', '--env', 'RUNNER_TEMP=/ci-temp',
            '--env', 'GITHUB_WORKSPACE=/workspace', '--env', 'CI=true']
    for name in PASS_ENV:
        if name in os.environ:
            args += ['--env', name + '=' + os.environ[name]]
    for name in ('GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY'):
        if name in os.environ:
            path = Path(os.environ[name]).resolve()
            require(path.is_relative_to(temporary), 'runner-confined step output paths')
            args += ['--env', name + '=/ci-temp/' + str(path.relative_to(temporary))]
    args += [execution_image, 'bash', '--noprofile', '--norc', '-e', '-o', 'pipefail',
             '/ci-temp/' + str(script.relative_to(temporary))]
    old_handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt('I received a CI cancellation signal.')
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, interrupted)
        docker(*args)
        actual = json.loads(docker('inspect', container))[0]
        require(actual['Image'] == execution_image and actual['HostConfig']['ReadonlyRootfs'],
                'the created container using the immutable recorded image')
        # I record the outer worker only as diagnostics; it does not supply tool/runtime bytes.
        print(json.dumps({'container': identity, 'outer_runner': {
            name: os.environ.get(name) for name in ('ImageOS', 'ImageVersion')}}, sort_keys=True), flush=True)
        return subprocess.run(['docker', 'start', '--attach', container]).returncode
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        subprocess.run(['docker', 'rm', '--force', container], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)
        receipt.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('build'); p.add_argument('archive', type=Path); p.add_argument('--platform', default='linux/amd64')
    p = sub.add_parser('load'); p.add_argument('archive', type=Path)
    p = sub.add_parser('shell'); p.add_argument('script', type=Path)
    args = parser.parse_args()
    try:
        if args.operation == 'build': build(args.archive, args.platform)
        elif args.operation == 'load': load(args.archive)
        else: return shell(args.script)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(str(error), file=sys.stderr); return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
