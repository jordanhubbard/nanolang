#!/usr/bin/env python3
"""I observe one cold production capture without changing supported tool versions."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.native_process import run_captured

OUT = ROOT / '.test_output/source-capture-diagnostic/linux-capture'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('I require the Linux capture boundary for this observation.')
    OUT.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which('cc')
    probe = ROOT / 'obj/test_module_generation_probe'
    helper = ROOT / 'bin/nano_as_capture.so'
    module = ROOT / 'modules/compiler_support'
    metadata = module / 'module.json'
    if not json.loads(metadata.read_text()).get('c_sources'):
        raise RuntimeError('I require the actual compiler support C inputs.')
    assembler_name = subprocess.check_output(
        [compiler, '-print-prog-name=as'], text=True, timeout=30).strip()
    assembler = shutil.which(assembler_name)
    if not assembler:
        raise RuntimeError('I require the actual selected assembler.')
    version = subprocess.check_output([assembler, '--version'], text=True, timeout=30)
    decision = subprocess.run([probe, 'assembler-version', version],
                              capture_output=True, text=True, timeout=30)
    record = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=ROOT, text=True).strip(),
        'compiler': str(Path(compiler).resolve()),
        'assembler': str(Path(assembler).resolve()),
        'assembler_version': version,
        'supported_version_decision': decision.returncode,
        'metadata_sha256': digest(metadata),
        'tool_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                       digest(p) for p in (Path(compiler).resolve(),
                                          Path(assembler).resolve(), probe, helper)},
        'limits': ['I observe one actual cold capture with unchanged production admission. '
                   'A retained refusal does not qualify source creation, snippets or release.'],
    }
    environment = dict(os.environ, NANO_BUILD_CACHE=str(OUT / 'cache'),
                       NANO_AS_CAPTURE_HELPER=str(helper),
                       NANO_TRACE_BUILD='1', NANO_VERBOSE_BUILD='1')
    command = [str(probe), 'build-info', str(module)]
    record['command'] = command
    record['environment'] = {name: environment[name] for name in
                             ('NANO_BUILD_CACHE', 'NANO_AS_CAPTURE_HELPER',
                              'NANO_TRACE_BUILD', 'NANO_VERBOSE_BUILD')}
    (OUT / 'observation.json').write_text(json.dumps(record, indent=2) + '\n')
    started = time.monotonic()
    try:
        result = run_captured(command, timeout=600, cwd=ROOT, env=environment)
        (OUT / 'capture.stdout').write_bytes(result.stdout)
        (OUT / 'capture.stderr').write_bytes(result.stderr)
        record['returncode'] = result.returncode
        if b'phase=build-metadata expected=1 observed=0' in result.stderr:
            raise RuntimeError('I did not reach the intended capture boundary.')
    except BaseException as error:
        record['observation_error'] = repr(error)
        raise
    finally:
        record['wall_seconds'] = time.monotonic() - started
        (OUT / 'observation.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
