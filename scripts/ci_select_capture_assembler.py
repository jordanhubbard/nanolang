#!/usr/bin/env python3
"""I provision a verified native assembler while preserving my configured compiler."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ci_qualify_as246 as process
from scripts import ci_build_capture_assembler as builder
from scripts.ci_capture_assembler_profile import (read_fixed_profile, retained_profile_metadata,
    materialize_retained_profile, RETAINED_MANIFEST_SHA256, RETAINED_AS_SHA256)
from scripts import ci_userguide_capture_diagnostics as diagnostics
from tests.native_process import run_captured


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compiler_identity():
    configured, origin = diagnostics.configured_compiler()
    argv = shlex.split(configured)
    if len(argv) != 1:
        raise RuntimeError('I require one literal configured CI compiler executable.')
    executable = shutil.which(argv[0])
    if not executable:
        raise RuntimeError('I cannot resolve my configured CI compiler.')
    path = str(Path(executable).resolve(strict=True))
    return argv, {'configured': configured, 'origin': origin, 'path': path, 'sha256': digest(path)}


def assembler_selection(compiler, environment, out, label):
    result = process.command([*compiler, '-print-prog-name=as'], label + '-selection',
                             30, env=environment, output=out)
    name = result.stdout.decode().strip()
    if not name or '\n' in name:
        raise RuntimeError('I require one actual selected assembler path.')
    executable = shutil.which(name, path=environment.get('PATH'))
    if not executable:
        raise RuntimeError('I cannot resolve the actual selected assembler.')
    path = Path(executable).resolve(strict=True)
    version = process.command([str(path), '--version'], label + '-version',
                              30, env=environment, output=out).stdout.decode()
    return path, version


def admitted(probe, version, out, label):
    result = run_captured([str(probe), 'assembler-version', version], timeout=30, cwd=ROOT)
    for suffix, content in (('.stdout', result.stdout), ('.stderr', result.stderr)):
        (out / (label + suffix)).write_bytes(content)
    if result.returncode not in (0, 1):
        raise RuntimeError('I require a successful native assembler-admission observation.')
    return result.returncode == 0


def settings_digest():
    names = ('CC', 'NANO_CC', 'CFLAGS', 'CPPFLAGS', 'LDFLAGS',
             'COMPILER_PATH', 'NANO_SHADOW_TIMEOUT_SECONDS', 'NANO_CAPTURE_TIMEOUT_MS')
    return hashlib.sha256(json.dumps({name: os.environ.get(name) for name in names},
                                    sort_keys=True).encode()).hexdigest()


def main():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('I select this GNU assembler profile only for native Linux CI.')
    environment_file = Path(os.environ['GITHUB_ENV'])
    out = ROOT / '.test_output/capture-assembler-ci'
    out.mkdir(parents=True, exist_ok=False)
    record = {'scope': 'I verify and select a source-backed CI assembler. Original native '
                      'capture, source and guide gates remain required after selection.'}
    try:
        before_settings = settings_digest()
        compiler, identity = compiler_identity()
        record['compiler_before'] = identity
        original = dict(os.environ)
        assembler, version = assembler_selection(compiler, original, out, 'original')
        record['original_assembler'] = {'path': str(assembler), 'sha256': digest(assembler),
                                        'version': version}
        probe = ROOT / 'obj/test_module_generation_probe'
        record['probe_sha256'] = digest(probe)
        record['capture_source_sha256'] = digest(ROOT / 'src/module_builder.c')
        supported = admitted(probe, version, out, 'original-admission')
        if not version.splitlines():
            raise RuntimeError('I require a nonempty actual assembler version report.')
        identified = version.splitlines()[0] == (
            'GNU assembler (' + builder.VENDOR + ') 2.46.0.20260210')
        if supported and not identified:
            if compiler_identity()[1] != identity or settings_digest() != before_settings:
                raise RuntimeError('I refuse changed original compiler settings during inspection.')
            record['selection'] = 'I preserve my already-supported original assembler.'
            record['compiler_after'] = compiler_identity()[1]
            record['settings_unchanged'] = True
            record['environment_changed'] = False
            return
        if not identified and not (version.startswith('GNU assembler (') and
                                    version.splitlines()[0].endswith(') 2.46')):
            raise RuntimeError('I require qualification of this unknown selected assembler family.')
        provided = original.get('NANO_CI_FIXED_ASSEMBLER_DIRECTORY', '')
        fixed = (Path(provided).resolve() if provided else
                 Path(original['RUNNER_TEMP']) / 'nanolang-capture-assembler-r1')
        producer_name = shutil.which('cc')
        if not producer_name:
            raise RuntimeError('I require the recorded native cc assembler producer.')
        producer = Path(producer_name).resolve(strict=True)
        producer_hash = digest(producer)
        retained = None
        if not provided:
            retained = retained_profile_metadata()
            if (platform.machine() != 'x86_64' or retained['compiler'] != str(producer) or
                    retained['compiler_sha256'] != producer_hash):
                raise RuntimeError('I require native qualification of this retained producer/toolchain binding.')
        if not fixed.exists():
            if provided:
                raise RuntimeError('I require the supplied completed assembler profile.')
            materialize_retained_profile(fixed)
        profile = read_fixed_profile(fixed)
        if profile['compiler'] != str(producer) or profile['compiler_sha256'] != producer_hash:
            raise RuntimeError('I require the actual recorded native assembler build compiler.')
        if retained is not None:
            if profile != retained or profile['assembler_sha256'] != RETAINED_AS_SHA256:
                raise RuntimeError('I refuse a default profile outside my pinned retained producer.')
            record['profile_origin'] = {'producer_run': 45009129801,
                'manifest': 'docs/evidence/capture-complete-hosted-20261122/manifest.json',
                'manifest_sha256': RETAINED_MANIFEST_SHA256,
                'scope': 'I reuse the exact qualified native/source/GAS/witness bytes; I rebuild or rerun no producer.'}
        execution = Path(original['RUNNER_TEMP']).resolve(strict=True) / 'nanolang-capture-selected-r1'
        if execution.is_relative_to(ROOT.resolve()):
            raise RuntimeError('I require native profile custody outside workspace clean trees.')
        if execution.is_symlink():
            raise RuntimeError('I refuse native profile custody through a directory symlink.')
        execution.mkdir(exist_ok=True)
        expected = execution / 'as'
        if expected.is_symlink():
            raise RuntimeError('I refuse native profile custody through an executable symlink.')
        shutil.copy2((fixed / 'bin/as').resolve(strict=True), expected)
        if digest(expected) != profile['assembler_sha256']:
            raise RuntimeError('I refuse changed native bytes after profile custody relocation.')
        expected.chmod(0o700)
        (out / 'validated-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
        old_search = original.get('COMPILER_PATH', '')
        search = str(expected.parent) + (os.pathsep + old_search if old_search else '')
        if '\n' in search or '\r' in search:
            raise RuntimeError('I refuse an invalid CI compiler search path.')
        selected_environment = dict(original, COMPILER_PATH=search)
        actual, actual_version = assembler_selection(compiler, selected_environment, out, 'fixed')
        if (actual != expected or digest(actual) != profile['assembler_sha256'] or
                actual_version != profile['assembler_version']):
            raise RuntimeError('I require the original compiler to select the verified fixed image.')
        shutil.copy2(actual, out / 'selected-as')
        if not admitted(probe, actual_version, out, 'fixed-admission'):
            raise RuntimeError('I refuse to export a profile absent from qualified production admission.')
        if compiler_identity()[1] != identity or settings_digest() != before_settings:
            raise RuntimeError('I refuse changed compiler, flags or shadow settings during provisioning.')
        with environment_file.open('a') as stream:
            stream.write('COMPILER_PATH=' + search + '\n')
        record['selection'] = 'I select the separately identified source-fixed GNU profile.'
        record['selected_assembler'] = {'path': str(actual), 'sha256': digest(actual),
            'version': actual_version, 'source_sha256': profile['source_sha256'],
            'fixed_source_sha256': profile['fixed_source_sha256'],
            'gas_cases': profile['gas_cases'], 'gas_expected_passes': profile['gas_expected_passes']}
        record['compiler_after'] = compiler_identity()[1]
        record['settings_unchanged'] = True
        record['environment_changed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        if 'error' not in record and 'environment_changed' in record and os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
                stream.write('fixed_profile=' + str(record['environment_changed']).lower() + '\n')


if __name__ == '__main__':
    if sys.argv[1:]:
        raise SystemExit('I require the configured CI environment without extra arguments.')
    main()
