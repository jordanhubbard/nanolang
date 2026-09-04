"""The one question every pre-release documentation check asks.

Each artifact -- README, CONTRIBUTING, the user guide, the deck -- is stale
under exactly the same condition: *what it describes changed since the last
release, and it did not*. Writing that once and parameterizing it by
(document, described paths) is why this module exists; the alternative is four
checkers that drift apart.

4.0 is the worked example. No commit touched CONTRIBUTING.md across the whole
span while `src/nanoisa/verifier*`, the Makefile gates and the optimization
policy all changed. A check of this shape would have said so on the first day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import repo
from .report import Finding


@dataclass(frozen=True)
class Watch:
    """A document, and the paths whose change implies it needs review."""

    document: str
    describes: list[str]
    why: str
    #: Cap on how many triggering paths to show. The point is to prove the
    #: claim, not to reprint a diff.
    sample: int = 4
    extra: dict = field(default_factory=dict)


def evaluate(watches: list[Watch], tag: str | None = None) -> list[Finding]:
    """Report every document whose subject moved while it stood still."""
    tag = tag or repo.latest_tag()
    if tag is None:
        # Before the first release there is no boundary to measure against,
        # and reporting every document as stale would be noise, not signal.
        return []

    changed = repo.changed_since(tag)
    findings: list[Finding] = []
    for watch in watches:
        if repo.matching(changed, [watch.document]):
            continue  # attended to in this release
        triggers = repo.matching(changed, watch.describes)
        if not triggers:
            continue  # subject did not move either; nothing to review
        findings.append(
            Finding(
                subject=watch.document,
                summary=f"{len(triggers)} change(s) since {tag} to what it documents, "
                f"but {watch.document} is untouched",
                detail=watch.why,
                evidence=sorted(triggers)[: watch.sample],
                evidence_total=len(triggers),
            )
        )
    return findings
