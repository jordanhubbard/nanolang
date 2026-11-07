#!/usr/bin/env python3
"""I qualify real image transport and tiny instrumented providers, not a bootstrap."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    temporary = Path(os.environ['RUNNER_TEMP']).resolve()
    temporary.mkdir(parents=True, exist_ok=True)
    driver = [sys.executable, str(ROOT / 'scripts/ci_container.py')]
    subprocess.run(driver + ['load', str(args.archive)], check=True, timeout=360)
    output = temporary / 'container-smoke-output'; output.write_text('')
    summary = temporary / 'container-smoke-summary'; summary.write_text('')
    environment = {**os.environ, 'GITHUB_OUTPUT': str(output), 'GITHUB_STEP_SUMMARY': str(summary), 'CFLAGS': '-O1 -g -fsanitize=address,undefined',
                   'LDFLAGS': '-fsanitize=address,undefined', 'ASAN_OPTIONS': 'detect_leaks=0'}
    producer = temporary / 'container-smoke-producer.sh'
    producer.write_text(r'''echo 'container-output=retained' >> "$GITHUB_OUTPUT"
echo 'I retained the container summary.' >> "$GITHUB_STEP_SUMMARY"
python3 -m unittest tests.test_ci_container tests.test_ci_bootstrap_checkpoint
python3 - <<'INNER'
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/workspace/scripts')
import ci_bootstrap_checkpoint as checkpoint
import yaml
jobs = yaml.safe_load(Path('/workspace/.github/workflows/ci.yml').read_text())['jobs']
for name in ('sanitizers-stage1', 'sanitizers', 'coverage-bootstrap', 'coverage'):
    job = jobs[name]
    assert 'instrumented-image' in job['needs']
    assert job['defaults']['run']['shell'] == 'python3 scripts/ci_container.py shell {0}'
    assert 'needs.instrumented-image.outputs.image-id' in job['env']['NANO_CI_IMAGE_ID']
    for step in job['steps']:
        if 'run' in step and 'shell' in step:
            assert step['shell'] == 'bash' and (step['name'].startswith('Require ') or step['name'] == 'Verify and load the immutable instrumented image')
        assert step['name'] not in ('Install dependencies', 'Install Python dependencies')
root = Path('/ci-temp/container-checkpoint-probe')
(root/'bin').mkdir(parents=True, exist_ok=True)
(root/'tiny.c').write_text('int main(void) { volatile int n = 7; return n == 7 ? 0 : 1; }\n')
subprocess.run(['cc','-O1','-g','-fsanitize=address,undefined',str(root/'tiny.c'),'-o',str(root/'bin/nanoc_stage1')],check=True)
(root/'bin/probe').write_bytes((root/'bin/nanoc_stage1').read_bytes()); (root/'bin/probe').chmod(0o755)
subprocess.run([str(root/'bin/probe')],check=True)
subprocess.run(['cc','--coverage',str(root/'tiny.c'),'-o',str(root/'coverage')],check=True)
subprocess.run([str(root/'coverage')],check=True)
assert list(root.glob('*.gcda'))
subprocess.run(['lcov','--capture','--directory',str(root),'--output-file',str(root/'coverage.info')],check=True)
checkpoint.ROOT = root
receipt = {'tools':['probe'],'hosts':[]}
identity = checkpoint.environment(receipt)
assert 'ImageVersion' not in identity['image']
manifest = checkpoint.write_archive(root,['bin/probe','bin/nanoc_stage1'],{'environment':identity},root/'checkpoint.tar.gz')
(root/'environment.json').write_text(json.dumps(identity,sort_keys=True))
assert not os.access('/usr/bin/cc',os.W_OK) or os.getuid() == 0
print('I compiled and executed sanitizer and coverage probes in the verified image.')
INNER
if touch /etc/nanolang-forbidden-write 2>/dev/null; then
  echo 'I require a read-only image root.' >&2
  exit 1
fi
''')
    consumer = temporary / 'container-smoke-consumer.sh'
    consumer.write_text(r'''python3 - <<'INNER'
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/workspace/scripts')
import ci_bootstrap_checkpoint as checkpoint
root = Path('/ci-temp/container-checkpoint-probe'); checkpoint.ROOT = root
manifest = checkpoint.read_archive(root/'checkpoint.tar.gz',root)
(root/'bin/probe').write_bytes(b'I replace only my disposable probe before restore.\n')
checkpoint.restore_files(root/'checkpoint.tar.gz',root,manifest)
receipt = {'tools':['probe'],'hosts':[]}
current = checkpoint.environment(receipt)
assert current == manifest['environment']
subprocess.run([str(root/'bin/probe')],check=True)
os.environ['CFLAGS'] += ' -DCHANGED_CHECKPOINT_FLAGS'
assert checkpoint.environment(receipt) != manifest['environment']
print('I restored exact providers in a second container despite a different outer runner label; changed flags still refuse identity.')
INNER
''')
    for version, script in [('20261012.334.1', producer), ('20261018.339.1', consumer)]:
        subprocess.run(driver + ['shell', str(script)], env={**environment, 'ImageVersion': version},
                       check=True, timeout=240)
    if output.read_text() != 'container-output=retained\n' or summary.read_text() != 'I retained the container summary.\n':
        raise RuntimeError('I require exact host-visible step output and summary mapping.')
    failure = temporary / 'container-smoke-failure.sh'; failure.write_text('exit 23\n')
    result = subprocess.run(driver + ['shell', str(failure)], env=environment, timeout=60)
    if result.returncode != 23:
        raise RuntimeError('I require the exact nonzero container exit status.')
    changed = {**environment, 'NANO_CI_ARCHIVE_SHA256': '0' * 64}
    result = subprocess.run(driver + ['load', str(args.archive)], env=changed, capture_output=True,
                            text=True, timeout=120)
    if result.returncode == 0 or 'unchanged image archive bytes' not in result.stderr:
        raise RuntimeError('I require a changed image archive identity to refuse before loading.')
    print('I passed real container transport, immutable-root, instrumented-provider and identity controls.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
