#!/usr/bin/env python3
"""
Validate module package metadata consistency.

Checks:
1) Any module with pkg_config entries must declare system_packages.
2) Every system_packages logical name must exist in packages.json.
3) Every used package must define apt/brew/pkg installers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_INSTALLERS = ("apt", "brew", "pkg")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    modules_root = repo_root / "modules"
    registry_path = repo_root / "packages.json"

    registry = load_json(registry_path)
    packages = registry.get("packages", {})
    if not isinstance(packages, dict):
        print("ERROR: packages.json is missing object field 'packages'")
        return 1

    issues: list[str] = []
    used_system_packages: set[str] = set()

    manifests = sorted(modules_root.rglob("module.json"))
    for manifest in manifests:
        rel = manifest.relative_to(repo_root)
        data = load_json(manifest)

        pkg_config = data.get("pkg_config", [])
        system_packages = data.get("system_packages", [])

        if pkg_config and not system_packages:
            issues.append(
                f"{rel}: has pkg_config={pkg_config} but missing system_packages"
            )

        for logical_name in system_packages:
            used_system_packages.add(logical_name)
            if logical_name not in packages:
                issues.append(
                    f"{rel}: system_packages contains unknown logical package '{logical_name}'"
                )

    for logical_name in sorted(used_system_packages):
        entry = packages.get(logical_name, {})
        install = entry.get("install", {}) if isinstance(entry, dict) else {}
        missing = [mgr for mgr in REQUIRED_INSTALLERS if mgr not in install]
        if missing:
            issues.append(
                f"packages.json:{logical_name}: missing install mappings for {', '.join(missing)}"
            )

    if issues:
        print("Module package audit FAILED:\n")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(
        f"Module package audit passed: {len(manifests)} manifests checked, "
        f"{len(used_system_packages)} logical packages validated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
