#!/usr/bin/env python3

import os
import re
from dataclasses import dataclass
from pathlib import Path


RE_MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


SKIP_DIRS = {
    ".git",
    ".factory",
    ".beads",
    "obj",
    "bin",
    "node_modules",
    "coverage-html",
}

# Archived snapshots are allowed to contain stale references.
SKIP_PATH_PREFIXES = {
    Path("planning/archive"),
}


@dataclass(frozen=True)
class BrokenLink:
    file_path: str
    target: str


def is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


def normalize_target(raw: str) -> str:
    target = raw.strip()

    # Remove optional title: (path "title")
    # Keep the first token unless the URL is wrapped in <...>
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()

    if " " in target and not is_external(target):
        target = target.split()[0]

    # Strip anchor
    if "#" in target and not target.startswith("#"):
        target = target.split("#", 1)[0]

    return target


def iter_markdown_files(repo_root: Path):
    for root, dirs, files in os.walk(repo_root):
        rel_root = Path(root).resolve().relative_to(repo_root)
        if any(rel_root == p or p in rel_root.parents for p in SKIP_PATH_PREFIXES):
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.lower().endswith(".md"):
                yield Path(root) / name


def find_broken_links_in_file(repo_root: Path, md_path: Path) -> list[BrokenLink]:
    rel_md_path = md_path.relative_to(repo_root)
    content = md_path.read_text(encoding="utf-8", errors="replace")

    # Links inside fenced code blocks are not rendered by markdown, so ignore them.
    in_fence = False
    stripped_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            stripped_lines.append(line)

    content = "".join(stripped_lines)

    broken: list[BrokenLink] = []
    for match in RE_MD_LINK.finditer(content):
        raw_target = match.group(1)
        target = normalize_target(raw_target)

        if not target or target.startswith("#"):
            continue
        if is_external(target):
            continue
        if any(ch in target for ch in ("*", "{", "}")):
            # Likely a template or glob; skip.
            continue

        # Resolve relative to the current file.
        if target.startswith("/"):
            abs_target = (repo_root / target.lstrip("/")).resolve()
        else:
            abs_target = (md_path.parent / target).resolve()

        if not abs_target.exists():
            broken.append(BrokenLink(str(rel_md_path), target))

    return broken


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    broken: list[BrokenLink] = []

    for md_path in iter_markdown_files(repo_root):
        broken.extend(find_broken_links_in_file(repo_root, md_path))

    if not broken:
        print("✓ markdown links: OK")
        return 0

    print("✗ broken markdown links found:\n")
    for b in broken:
        print(f"- {b.file_path}: {b.target}")

    print(f"\nTotal: {len(broken)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
