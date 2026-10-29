#!/usr/bin/env python3
"""I author my retained-project test policy, command profile and input manifest for Literate AI planning.

I must run under the Literate AI interpreter (~/.local/share/literate-ai/venv/bin/python) because I use its
typed contracts to compute the hashes and identities it will later recompute. I read the case list from
make's own prerequisite graph for `test`, never from a hand-kept list. I refuse an unclean tree: the framework
walks the file system, so ignored or untracked files under a retained root would enter the manifest.
My output is review input. It is not a receipt, and it grants no execution authority."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from literate_ai.adapters.retained_project_inputs import discover_retained_project_inputs
from literate_ai.cli.execution_workers import select_execution_worker
from literate_ai.contracts.identity import ContentIdentity, canonical_identity
from literate_ai.contracts.retained_project import RetainedProjectLimits
from literate_ai.contracts.retained_project_acceptance import RetainedProjectAcceptanceCase, RetainedProjectAcceptancePolicy
from literate_ai.contracts.retained_project_lifecycle import RetainedProjectAction, RetainedProjectProfile
from literate_ai.contracts.retained_project_tests import RetainedProjectExpectedCase, RetainedProjectTestPolicy
from literate_ai.contracts.testing import ProjectTestReceiptPolicy

ROOT = Path(__file__).resolve().parents[1]
RETAINED = 'literate/retained'
POLICY = f'{RETAINED}/test-policy.json'
PROFILE = f'{RETAINED}/profile.json'
MANIFEST = 'literate/verification/retained-manifest.json'
# Tracked native binaries are stale build outputs (the Makefile regenerates them), so they are not inputs. The framework
# inspects every Mach-O input and refuses an ASan-linked one, so I leave them out and record exactly what I left out.
EXCLUDED = 'literate/retained/excluded-native-artifacts.json'
MACHO = {b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xce', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
         b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca', b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca'}
# The verifier-owned acceptance policy must sit outside every manifest root (the framework withholds it).
ACCEPTANCE = 'verification/retained-project-acceptance.json'
# Built executables whose exact exit, stdout and stderr I record. I only pick output that carries no build
# time, absolute path or argv[0]: `nanoc_c --version` embeds a build timestamp, so it cannot be a case.
ACCEPTANCE_CASES = (
    ('nano-vm-rejects-unknown-option', 'bin/nano_vm', ('--bogus',)),
    ('nanoisa-emitter-entry-assertions', 'bin/nanoisa_emitter', ()),
    ('parser-entry-assertions', 'bin/parser', ()),
    ('nvm2c-usage', 'bin/nvm2c', ()),
)
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


def component_locks():
    """The project's current component-lock tuple, read through the framework API the planner itself uses."""
    from literate_ai.adapters.component_lock_application import current_project_component_locks
    from literate_ai.projects import load_project
    return tuple(lock.identity for lock in current_project_component_locks(load_project(ROOT)))


def observe(executable, argv):
    """Run one built executable twice from a neutral directory with an empty environment; refuse unstable output."""
    import tempfile
    seen = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([str(ROOT / executable), *argv], cwd=cwd, env={}, input=b'', capture_output=True, timeout=60)
        seen.append((result.returncode, result.stdout, result.stderr))
    if seen[0] != seen[1] or str(ROOT).encode() in seen[0][1] + seen[0][2]:
        sys.exit(f'{executable} {argv}: output is not deterministic or embeds the tree path; it cannot be an acceptance case')
    return seen[0]


def write_acceptance():
    cases = []
    for case_id, executable, argv in ACCEPTANCE_CASES:
        if not (ROOT / executable).is_file():
            sys.exit(f'{executable} is not built; run `make build` first so I can record its exact output')
        code, out, err = observe(executable, argv)
        cases.append(RetainedProjectAcceptanceCase(case_id, 'accept', executable, argv, b'', code, out, err, 60))
    policy = RetainedProjectAcceptancePolicy(component_locks(), tuple(cases))
    target = ROOT / ACCEPTANCE
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + '\n')
    return tuple(c[0] for c in ACCEPTANCE_CASES), digest(ACCEPTANCE)


def tracked_native():
    """Tracked regular files that are Mach-O, other than the declared retained tools."""
    tools = {f'{RETAINED}/tools/{t}' for t in TOOLS}
    found = []
    for name in run('git', 'ls-files', '-z').split('\0'):
        path = ROOT / name
        if not name or name in tools or path.is_symlink() or not path.is_file():
            continue
        with path.open('rb') as handle:
            if handle.read(4) in MACHO:
                found.append(name)
    return sorted(found)


def expand_roles(roles, excluded):
    """Replace a root that contains an excluded file by its remaining children, so the walk never sees it."""
    skip = set(excluded)
    result = {}

    def visit(path, role):
        if path in skip:
            return
        if (ROOT / path).is_dir() and any(e.startswith(path + '/') for e in skip):
            for child in sorted(os.listdir(ROOT / path)):
                visit(f'{path}/{child}', role)
        else:
            result[path] = role

    for path, role in roles.items():
        visit(path, role)
    return result


def digest(path):
    import hashlib
    return ContentIdentity.parse_uri('sha256:' + hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default='host')
    args = parser.parse_args()
    require_clean()
    excluded = tracked_native()
    roles = expand_roles(ROLES, excluded)
    (ROOT / EXCLUDED).write_text(json.dumps({'schema': 'nanolang/excluded-native-artifacts@1', 'reason': 'tracked native binaries are '
                                              'stale build outputs; the Makefile regenerates them', 'paths': excluded}, indent=2) + '\n')
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
    accept_ids, acceptance_identity = write_acceptance()
    actions = (
        RetainedProjectAction('build', 'build', 'build-native', ('{input_root}', '{output_root}'), (), ('native-build',), ('bin',), 21600),
        RetainedProjectAction('tests', 'test', 'run-cases', (POLICY, '{output_root}/test-result.json', 'tests'), ('build',),
                              tuple(cases), ('test-result.json',), 86400),
        RetainedProjectAction('docs', 'docs', 'host-make', ('userguide-export',), ('build',), ('userguide-export',), (), 7200),
        RetainedProjectAction('accept', 'accept', 'host-make', ('release-docs-check',), ('tests', 'docs'),
                              accept_ids, (), 7200),
    )
    limits = RetainedProjectLimits(1_000_000, 512 * 1024 * 1024, 16 * 1024 * 1024 * 1024)
    executable = tuple(f'{RETAINED}/tools/{t}' for t in TOOLS)
    zero = canonical_identity('pending')

    def profile(toolchain, dependency):
        return RetainedProjectProfile(
            args.target, worker.worker.identity, toolchain, dependency, digest(POLICY), digest('literate.release.json'),
            tuple((t, f'{RETAINED}/tools/{t}', digest(f'{RETAINED}/tools/{t}')) for t in TOOLS),
            actions, ('bin',), acceptance_identity,
        )

    def write_profile(p):
        (ROOT / PROFILE).write_text(json.dumps(p.to_dict(), indent=2, sort_keys=True) + '\n')

    write_profile(profile(zero, zero))
    manifest = discover_retained_project_inputs(ROOT, roles, limits, executable_paths=executable)
    final = profile(manifest.role_identity({'sdk', 'toolchain'}), manifest.role_identity({'dependency'}))
    write_profile(final)
    manifest = discover_retained_project_inputs(ROOT, roles, limits, executable_paths=executable)
    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / MANIFEST).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + '\n')
    print(json.dumps({'cases': len(cases), 'members': len(manifest.members), 'directories': len(manifest.directories),
                      'manifest_identity': manifest.identity.uri, 'worker_identity': worker.worker.identity.uri}, sort_keys=True))


if __name__ == '__main__':
    main()
