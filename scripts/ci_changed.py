#!/usr/bin/env python3
"""I classify the files a push changed so CI can skip its long jobs for documentation-only pushes.

I read changed paths on stdin and print `code=true` or `code=false`. I answer `true` whenever I am unsure: an
empty list, an unreadable path or any file outside the documentation patterns keeps the long jobs running."""
import fnmatch
import argparse
import json
import os
import re
import subprocess
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


def github_event_changes(event_name, event, cwd=None):
    """I compare the full PR range, or the actual push range, using real Git."""
    if event_name == 'pull_request':
        request = event.get('pull_request', {})
        before, after = request.get('base', {}).get('sha'), request.get('head', {}).get('sha')
    elif event_name == 'push':
        before, after = event.get('before'), event.get('after')
    else:
        return True
    if any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{40}', value)
           or value == '0' * 40 for value in (before, after)):
        return True
    try:
        fetched = subprocess.run(['git', 'fetch', '--no-tags', '--depth=1', 'origin', before, after],
                                 cwd=cwd, capture_output=True, timeout=60)
        if fetched.returncode:
            return True
        changed = subprocess.run(['git', 'diff', '--name-only', '-z', before, after, '--'],
                                 cwd=cwd, capture_output=True, timeout=30)
        if changed.returncode:
            return True
        return classify(changed.stdout.decode('utf-8', errors='surrogateescape').split('\0'))
    except (OSError, subprocess.SubprocessError):
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--github-event', action='store_true',
                        help='I read the actual GitHub event and fail closed on missing Git evidence.')
    args = parser.parse_args()
    if args.github_event:
        try:
            with open(os.environ['GITHUB_EVENT_PATH']) as source:
                event = json.load(source)
            code = github_event_changes(os.environ.get('GITHUB_EVENT_NAME'), event)
        except (KeyError, OSError, ValueError, AttributeError, TypeError):
            code = True
    else:
        code = classify(sys.stdin.read().splitlines())
    print('code=' + ('true' if code else 'false'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
