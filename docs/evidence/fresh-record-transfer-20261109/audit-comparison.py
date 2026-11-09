"""I audit all six retained creations without running a new factory."""
import argparse
import functools
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PIN = 'a469d94ec3830882a573186743d2647b9fde1a3f'
BASE = '52933101c15be6fd189c5cecc2189cbbb29a23ba'


def digest(data):
    return hashlib.sha256(data).hexdigest()


@functools.cache
def git_digest(pin, path):
    return digest(subprocess.check_output(['git', 'show', pin+':'+path], cwd=ROOT))


def audit(archive, output, retained):
    assert not output.exists() and not retained.exists()
    parser = ROOT/'scripts/compare_cseed_source_creation.py'
    assert digest(parser.read_bytes()) == git_digest(PIN, 'scripts/compare_cseed_source_creation.py')
    spec = importlib.util.spec_from_file_location('comparison', parser)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    inventory, texts, native, findings = [], {}, [], []
    output.mkdir(parents=True)
    retained.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names)) == 55
        for info in z.infolist():
            assert not info.filename.startswith('/') and '..' not in Path(info.filename).parts
            data = z.read(info)
            inventory.append(dict(path=info.filename, bytes=len(data), sha256=digest(data)))
            if data[:4] == b'\x7fELF':
                kind = struct.unpack_from('<H' if data[5] == 1 else '>H', data, 16)[0]
                assert data[5] in (1, 2) and kind in (2, 3)
                target = retained/(info.filename+'.gz')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gzip.compress(data, mtime=0))
                assert digest(gzip.decompress(target.read_bytes())) == digest(data)
                native.append(dict(path=info.filename, bytes=len(data), sha256=digest(data),
                                   kind='ELF executable/PIE', retained=str(target)))
            else:
                texts[info.filename] = data.decode()
        prefix = 'cseed-source-comparison/'
        load = lambda name: json.loads(z.read(prefix+name))
        plan, summary = load('plan.json'), load('summary.json')
        assert plan['baseline'] == BASE and plan['candidate'] == PIN
        assert plan['order'] == ['original', 'modified', 'modified', 'original', 'original', 'modified']
        assert plan['shadow_timeout_seconds'] == 10 and plan['native_build_budget'] == 600
        assert 'shadow_phase_observation' not in plan and 'process_observation' not in plan
        assert summary['comparison_valid'] and summary['all_planned_samples_finished']
        assert not summary['all_creations_passed'] and len(summary['samples']) == 6
        providers = load('providers-before.json')
        assert providers == load('providers-after.json')
        for variant in ('original', 'modified'):
            assert digest(z.read(prefix+'providers/'+variant+'/nanoc_c')) == providers[variant]['bin/nanoc_c']
        shared = load('shared-inputs.json')
        assert len(shared) == 197
        for path, value in shared.items():
            assert git_digest(BASE, path) == git_digest(PIN, path) == value, path
        observed, full = [], None
        for sample in summary['samples']:
            index, variant = sample['index'], sample['variant']
            succeeded = index in (1, 5)
            assert variant == plan['order'][index-1] and sample['creation_succeeded'] == succeeded
            assert sample['process']['returncode'] == (0 if succeeded else 1)
            sample_root = prefix+'samples/'+f'{index:02}-{variant}/'
            identity = json.loads(z.read(sample_root+'inputs.json'))
            assert identity['before'] == identity['after'] and len(identity['before']) == 67
            for path, value in identity['before'].items():
                relative = Path(path).relative_to(plan['worktrees'][variant]).as_posix()
                expected = (providers[variant]['bin/nanoc_c'] if relative == 'bin/nanoc_c'
                            else git_digest(BASE if variant == 'original' else PIN, relative))
                assert value == expected, path
            raw = z.read(sample_root+'source-build.stderr').decode()
            trace = json.loads(json.dumps(comparison.shadow_trace(raw, Path(plan['worktrees'][variant]))))
            assert trace == sample['trace'] and trace['failures'] == 0
            observed, full = comparison.check_selection(observed, full, trace['selection_prefix'], succeeded)
            if succeeded:
                assert trace['starts'] == trace['completes'] == 1286
                assert identity['product_before_run'] == identity['product_after_run'] == digest(z.read(sample_root+'compiler'))
            else:
                assert identity['product_before_run'] is identity['product_after_run'] is None
                assert sample_root+'compiler' not in names
                assert 'I stopped shadow execution after 10 seconds.' in raw
            findings.append(dict(index=index, variant=variant, creation_succeeded=succeeded,
                                 starts=trace['starts'], completes=trace['completes'],
                                 failures=trace['failures'], summed_body_seconds=trace['summed_body_seconds'],
                                 process=sample['process'], source_compiler_executed=False))
        assert len(observed) == 1286 and len(native) == 4
    for name, value in [('inventory', inventory), ('texts', texts)]:
        (output/(name+'.json.gz')).write_bytes(gzip.compress(json.dumps(value, sort_keys=True).encode(), mtime=0))
    report = dict(run=42598208925, artifact=13665163163, source_pin=PIN, baseline=BASE,
                  archive_sha256=digest(archive.read_bytes()), entries_crc_hash_checked=len(inventory),
                  shared_git_matched_inputs=197, stable_git_or_provider_inputs_each=67,
                  native_images=native, unique_native_image_hashes=len({n['sha256'] for n in native}),
                  samples=findings, baseline_creations=2, candidate_creations=0,
                  full_successful_pairs=0, all_planned_samples_finished=True,
                  comparison_protocol_valid=True, optimization_accepted=False,
                  actual_exit_teardown_cause_established=False,
                  complete_provider_sdk_bytes_retained=False, release_qualified=False,
                  scope='I retain two real C seed images and two baseline-created native compilers. Other provider/tool hashes are reports without retained bytes. No complete successful pair supports a whole-factory gain; I reject adoption and do not rerun this completed plan.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(entries=len(inventory), native_images=len(native),
                          baseline_creations=2, candidate_creations=0)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('retained', type=Path)
    a = p.parse_args()
    audit(a.archive, a.output, a.retained)
