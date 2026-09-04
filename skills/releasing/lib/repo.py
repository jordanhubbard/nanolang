"""Git and path resolution shared by every release check.

Every helper under `skills/releasing/` needs the same three facts: where the
repository root is, what the last released tag was, and which files changed
since it. They are here rather than in each checker so that a change to how a
release boundary is determined lands in one place.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from functools import lru_cache
from pathlib import Path


class RepoError(RuntimeError):
    """A git query failed in a way the caller cannot sensibly continue past."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), capture_output=True, text=True, cwd=str(repo_root())
    )
    if result.returncode != 0:
        raise RepoError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


@lru_cache(maxsize=1)
def repo_root() -> Path:
    result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
    )
    if result.returncode != 0:
        raise RepoError("not inside a git repository")
    return Path(result.stdout.strip())


@lru_cache(maxsize=1)
def latest_tag() -> str | None:
    """The release boundary to measure against.

    `RELEASE_BASE_TAG` overrides it. That exists so these gates can be pointed
    at a past release and shown to catch what actually went stale there --
    a gate nobody has watched fail is not evidence of anything.
    """
    override = os.environ.get("RELEASE_BASE_TAG", "").strip()
    if override:
        return override
    try:
        tags = _git("tag", "-l", "v*", "--sort=-v:refname").splitlines()
    except RepoError:
        return None
    return tags[0] if tags else None


def head_ref() -> str:
    """The end of the range being examined.

    `RELEASE_HEAD_REF` overrides it, which is what makes these gates testable:
    pointing base and head at two shipped tags replays a past release exactly
    as it was, rather than as the current working tree has since repaired it.
    """
    return os.environ.get("RELEASE_HEAD_REF", "").strip() or "HEAD"


@lru_cache(maxsize=None)
def changed_since(tag: str) -> tuple[str, ...]:
    """Repository-relative paths touched between `tag` and the head ref.

    When the head is the working tree, uncommitted edits count too. A release
    is prepared with edits in flight, so a document already modified but not
    yet committed has been attended to and must not be reported as neglected.
    """
    head = head_ref()
    committed = _git("diff", "--name-only", f"{tag}..{head}").splitlines()
    working: list[str] = []
    if head == "HEAD":
        uncommitted = _git("status", "--porcelain").splitlines()
        working = [line[3:].split(" -> ")[-1] for line in uncommitted if len(line) > 3]
    return tuple(sorted({p for p in (*committed, *working) if p}))


def matching(paths: tuple[str, ...], patterns: list[str]) -> list[str]:
    """Paths matching any glob in `patterns`.

    A pattern without a wildcard matches by prefix, so `src/nanoisa` covers
    everything beneath it without each caller writing `src/nanoisa/**`.
    """
    hits: list[str] = []
    for path in paths:
        for pattern in patterns:
            wild = any(ch in pattern for ch in "*?[")
            if (wild and fnmatch.fnmatch(path, pattern)) or (
                not wild and (path == pattern or path.startswith(pattern.rstrip("/") + "/"))
            ):
                hits.append(path)
                break
    return hits
