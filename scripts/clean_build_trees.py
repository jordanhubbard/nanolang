#!/usr/bin/env python3
"""I remove compiler build trees without collecting retained runtime caches."""
import argparse
import os
from pathlib import Path
import shutil


def clean(roots, caches, workspace):
    workspace = Path(os.path.abspath(workspace))
    roots = [Path(os.path.abspath(root)) for root in roots]
    caches = [Path(os.path.abspath(cache)) for cache in caches if cache]
    caches += [Path(os.path.realpath(cache)) for cache in caches]
    # I validate the whole request before removing anything. I do not accept
    # workspace/home/filesystem roots or follow directory symlinks while walking.
    for root in roots:
        if root in (workspace, Path.home(), Path(root.anchor)) or not root.is_relative_to(workspace):
            raise ValueError(f"I only clean build trees strictly inside {workspace}: {root}")
        for parent in root.parents:
            if parent == workspace: break
            if parent.is_symlink():
                raise ValueError(f"I refuse a cleanup path through a symlink: {root}")
    # I also retain earlier configured cache locations. These names are only
    # conservative retention hints, never evidence authorizing collection.
    def scan_error(error):
        raise error
    for root in roots:
        if root.is_symlink() or not root.is_dir(): continue
        for base, directories, files in os.walk(root, followlinks=False, onerror=scan_error):
            path = Path(base)
            if any(path == cache or path.is_relative_to(cache) for cache in caches):
                directories[:] = []
            elif path.name.startswith((".nano-gen-", ".nano-build-")) or ".build.lock" in files or any(
                    name.startswith((".nano-gen-", ".nano-build-")) for name in directories + files):
                caches.append(path)
                directories[:] = []
    def remove(path):
        if any(path == cache or path.is_relative_to(cache) for cache in caches):
            return
        contains_cache = any(cache.is_relative_to(path) for cache in caches)
        if path.is_symlink():
            if not contains_cache: path.unlink()
        elif path.is_dir():
            if contains_cache:
                for entry in path.iterdir(): remove(entry)
            else:
                shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for root in roots: remove(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--file", action="append", default=[])
    args = parser.parse_args()
    try:
        files = [Path(path) for path in args.file]
        files += [path for pattern in args.glob for path in Path.cwd().glob(pattern)]
        roots = args.root + [str(path) for path in files if not path.is_dir() or path.is_symlink()]
        clean(roots, args.cache, Path.cwd())
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
