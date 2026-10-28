#!/usr/bin/env python3
"""I author my retained-project test policy, command profile and input manifest for Literate AI planning.

I must run under the Literate AI interpreter (~/.local/share/literate-ai/venv/bin/python) because I use its
typed contracts to compute the hashes and identities it will later recompute. I read the case list from
make's own prerequisite graph for `test`, never from a hand-kept list. I refuse an unclean tree: the framework
walks the file system, so ignored or untracked files under a retained root would enter the manifest.
My output is review input. It is not a receipt, and it grants no execution authority."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from literate_ai.adapters.retained_project_inputs import discover_retained_project_inputs
from literate_ai.cli.execution_workers import select_execution_worker
from literate_ai.contracts.identity import ContentIdentity, canonical_identity
from literate_ai.contracts.retained_project import RetainedProjectLimits
from literate_ai.contracts.retained_project_lifecycle import RetainedProjectAction, RetainedProjectProfile
from literate_ai.contracts.retained_project_tests import RetainedProjectExpectedCase, RetainedProjectTestPolicy
from literate_ai.contracts.testing import ProjectTestReceiptPolicy

ROOT = Path(__file__).resolve().parents[1]
RETAINED = 'literate/retained'
POLICY = f'{RETAINED}/test-policy.json'
PROFILE = f'{RETAINED}/profile.json'
MANIFEST = 'literate/verification/retained-manifest.json'
TOOLS = ('build-native', 'host-make', 'run-cases')
# Roots and the ownership role each carries. Everything under a root takes its role.
ROLES = {
    'src': 'source', 'src_nano': 'source', 'modules': 'source', 'std': 'source', 'stdlib': 'source',
    'include': 'source', 'schema': 'source', 'catalogs': 'source',
    'Makefile': 'build-definition', 'Makefile.gnu': 'build-definition', 'GNUmakefile': 'build-definition',
    'scripts': 'build-definition', PROFILE: 'build-definition',
    'tests': 'test-definition', 'formal': 'test-definition',
    'docs': 'docs-definition', 'userguide': 'docs-definition', 'README.md': 'docs-definition',
    f'{RETAINED}/tools': 'toolchain', POLICY: 'test-policy', 'literate.release.json': 'release-policy',
}


def run(*args, check=True):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=check).stdout


def test_cases():
    """The transitive prerequisites of `test` and its `test-impl` sub-make, from make's database."""
    graph = {}
    for line in run('make', '-pn', '-q', 'test-impl', check=False).splitlines():
        match = re.match(r'^([A-Za-z0-9_.-]+):(?!=)\s*(.*)$', line)
        if match and not line.startswith(('#', '\t')):
            graph.setdefault(match.group(1), set()).update(match.group(2).split())
    seen, stack = set(), ['test-impl', 'test']
    while stack:
        for dep in graph.get(stack.pop(), ()):
            if dep not in seen and re.fullmatch(r'[A-Za-z0-9_.-]+', dep):
                seen.add(dep)
                stack.append(dep)
    cases = sorted(c for c in seen if c.startswith('test-') or c == 'shadow-check')
    if not cases:
        sys.exit('no test cases found in make test prerequisites')
    return cases


def require_clean():
    generated = (POLICY, PROFILE, MANIFEST)
    dirty = run('git', 'status', '--porcelain', '--ignored', '--', *[r for r in ROLES if (ROOT / r).exists()])
    dirty = '\n'.join(l for l in dirty.splitlines() if not l[3:].startswith(generated))
    if dirty.strip():
        sys.exit('retained roots must be clean and tracked; found:\n' + dirty[:600])


def digest(path):
    import hashlib
    return ContentIdentity.parse_uri('sha256:' + hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default='host')
    args = parser.parse_args()
    require_clean()
    cases = test_cases()
    receipt = json.loads((ROOT / 'literate.project.json').read_text()).get('test_receipt_policy')
    if receipt is None:
        sys.exit('literate.project.json has no test_receipt_policy')
    configured = ProjectTestReceiptPolicy.from_dict(receipt)
    policy = RetainedProjectTestPolicy(
        configured.identity,
        tuple(RetainedProjectExpectedCase(c, 'tests') for c in cases),
        (('tests', 'test-result.json'),),
    )
    (ROOT / POLICY).write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + '\n')
    # The profile must be a manifest member, and it names the manifest's role closures; so I write it
    # with placeholder closure identities, discover the manifest, then write the real profile and re-discover.
    worker = select_execution_worker(argparse.Namespace(), project_root=ROOT, target_profile=args.target)
    actions = (
        RetainedProjectAction('build', 'build', 'build-native', ('{output_root}',), (), ('native-build',), ('bin',), 21600),
        RetainedProjectAction('tests', 'test', 'run-cases', (POLICY, '{output_root}/test-result.json', 'tests'), ('build',),
                              tuple(cases), ('test-result.json',), 86400),
        RetainedProjectAction('docs', 'docs', 'host-make', ('userguide-export',), ('build',), ('userguide-export',), (), 7200),
        RetainedProjectAction('accept', 'accept', 'host-make', ('release-docs-check',), ('tests', 'docs'),
                              ('release-docs-check',), (), 7200),
    )
    limits = RetainedProjectLimits(1_000_000, 512 * 1024 * 1024, 16 * 1024 * 1024 * 1024)
    executable = tuple(f'{RETAINED}/tools/{t}' for t in TOOLS)
    zero = canonical_identity('pending')

    def profile(toolchain, dependency):
        return RetainedProjectProfile(
            args.target, worker.worker.identity, toolchain, dependency, digest(POLICY), digest('literate.release.json'),
            tuple((t, f'{RETAINED}/tools/{t}', digest(f'{RETAINED}/tools/{t}')) for t in TOOLS),
            actions, ('bin',),
        )

    def write_profile(p):
        (ROOT / PROFILE).write_text(json.dumps(p.to_dict(), indent=2, sort_keys=True) + '\n')

    write_profile(profile(zero, zero))
    manifest = discover_retained_project_inputs(ROOT, ROLES, limits, executable_paths=executable)
    final = profile(manifest.role_identity({'sdk', 'toolchain'}), manifest.role_identity({'dependency'}))
    write_profile(final)
    manifest = discover_retained_project_inputs(ROOT, ROLES, limits, executable_paths=executable)
    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / MANIFEST).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + '\n')
    print(json.dumps({'cases': len(cases), 'members': len(manifest.members), 'directories': len(manifest.directories),
                      'manifest_identity': manifest.identity.uri, 'worker_identity': worker.worker.identity.uri}, sort_keys=True))


if __name__ == '__main__':
    main()
