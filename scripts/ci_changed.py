#!/usr/bin/env python3
"""I classify the files a push changed so CI can skip its long jobs for documentation-only pushes.

I read changed paths on stdin and print `code=true` or `code=false`. I answer `true` whenever I am unsure: an
empty list, an unreadable path or any file outside the documentation patterns keeps the long jobs running."""
import fnmatch
import sys

# I list only paths whose change cannot alter a built tool or a test input. Documentation that a test reads
# (the roadmap, release notes) still counts as documentation here; my fast gate and docs link job always run.
DOCUMENTATION = ('docs/*.md', 'docs/*.txt', '*.md', 'LICENSE')


def is_documentation(path):
    path = path.strip().lstrip('./')
    return bool(path) and any(fnmatch.fnmatch(path, pattern) for pattern in DOCUMENTATION)


def classify(paths):
    paths = [path for path in (line.strip() for line in paths) if path]
    if not paths:
        return True
    return not all(is_documentation(path) for path in paths)


def main():
    print('code=' + ('true' if classify(sys.stdin.read().splitlines()) else 'false'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
