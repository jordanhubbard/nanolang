#!/usr/bin/env python3
"""Check my original Git source custody without executing native build actions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def owner(path: str, policy: dict) -> str:
    matches = []
    file_prefixes = set(policy.get('file_prefixes', []))
    for name, prefixes in policy['owners'].items():
        for prefix in prefixes:
            prefix_rule = prefix.endswith('/') or prefix in file_prefixes
            if (prefix_rule and path.startswith(prefix)) or path == prefix:
                matches.append((len(prefix), name))
    if not matches:
        return policy['fallback_owner']
    length = max(n for n, _ in matches)
    names = {name for n, name in matches if n == length}
    if len(names) != 1:
        raise ValueError(f'ambiguous preservation owner: {path}')
    return names.pop()


def records(repo: Path, revision: str) -> dict[str, tuple[str, str, str]]:
    raw = subprocess.run(['git', 'ls-tree', '-rz', '--full-tree', revision],
                         cwd=repo, capture_output=True, check=True, timeout=30).stdout
    result = {}
    for item in raw.split(b'\0'):
        if not item:
            continue
        header, path = item.split(b'\t', 1)
        mode, kind, oid = header.decode('ascii').split()
        result[path.decode('utf-8')] = (mode, kind, oid)
    return result


def inspect(repo: Path, policy: dict, revision: str) -> dict:
    revision = subprocess.run(['git', 'rev-parse', '--verify', revision + '^{commit}'],
                              cwd=repo, capture_output=True, check=True, timeout=10).stdout.decode().strip()
    baseline = records(repo, policy['baseline_commit'])
    current = records(repo, revision)
    counts: dict[str, int] = {}
    changed = []
    allowed = set(policy['reviewed_metadata_changes'])
    digest = hashlib.sha256()
    for path, entry in sorted(baseline.items()):
        name = owner(path, policy)
        counts[name] = counts.get(name, 0) + 1
        digest.update(json.dumps([path, *entry, name], ensure_ascii=True,
                                 separators=(',', ':')).encode() + b'\n')
        if current.get(path) != entry and path not in allowed:
            changed.append(path)
    # Git tree comparison alone cannot qualify an edited working tree.
    raw = subprocess.run(['git', 'diff', '--name-only', '-z', revision, '--'],
                         cwd=repo, capture_output=True, check=True, timeout=30).stdout
    dirty = [p.decode() for p in raw.split(b'\0') if p and p.decode() in baseline
             and p.decode() not in allowed]
    return {'schema': 'nanolang/source-preservation-report@1',
            'baseline_commit': policy['baseline_commit'], 'revision': revision,
            'original_entries': len(baseline), 'owners': counts,
            'inventory_sha256': digest.hexdigest(),
            'inventory_format': 'sorted JSONL [path, Git mode, Git type, Git object ID, preservation owner]',
            'changed_original_paths': changed, 'dirty_original_paths': dirty,
            'preserved': not changed and not dirty,
            'native_lifecycle_qualified': False,
            'limits': 'Git custody only; original Git object IDs are not file SHA256 or native SBOM/toolchain admission'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--revision', default='HEAD')
    args = parser.parse_args()
    repo = args.project.resolve()
    policy = json.loads((repo / 'literate/source-ownership.json').read_text())
    report = inspect(repo, policy, args.revision)
    print(json.dumps(report, indent=2))
    return 0 if report['preserved'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
