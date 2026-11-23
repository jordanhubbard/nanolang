#!/usr/bin/env python3
"""I reuse an unchanged qualified assembler producer for a fresh complete GCC consumer."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ci_fixed_clang_inputs as shared


def require_compiler(profile, expected_hash):
    compiler = Path(shutil.which('cc')).resolve(strict=True)
    metadata = json.loads((profile / 'build.json').read_text())
    if str(compiler) != metadata['compiler'] or shared.digest(compiler) != expected_hash:
        raise RuntimeError('I require the unchanged actual qualified GCC producer compiler.')
    return {'path': str(compiler), 'sha256': expected_hash}


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('I prepare this consumer only in its isolated GitHub Actions checkout.')
    out = ROOT / '.test_output/source-capture-diagnostic/fixed-gcc-inputs'
    out.mkdir(parents=True, exist_ok=False)
    record = {'scope': 'I prepare a fresh complete GCC consumer, preserve its original compiler, '
                      'and repeat no already-qualified assembler source build or GAS cohort.'}
    try:
        prior = shared.prepare_prior(out, record)
        original = prior / 'fixed-as246'
        record['compiler'] = require_compiler(original, record['producer']['producer_compiler_sha256'])
        target = ROOT / '.test_output/source-capture-diagnostic/fixed-as246'
        if target.exists():
            raise RuntimeError('I refuse to overwrite an existing native assembler profile.')
        before = shared.read_fixed_profile(original)
        original.rename(target)
        if shared.read_fixed_profile(target) != before:
            raise RuntimeError('I refuse changed producer metadata after artifact layout relocation.')
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            stream.write('NANO_FIXED_AS246_DIRECTORY=' + str(target) + '\n')
        record.update(prepared=True, assembler_sha256=before['assembler_sha256'],
                      profile_relocated_from=str(original), profile=str(target),
                      compiler_unchanged=True)
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
