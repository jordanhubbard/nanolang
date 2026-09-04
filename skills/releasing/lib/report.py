"""One output shape for every release check.

A releaser reads these during a long automated run, so each finding has to say
what is wrong, why it matters, and what would settle it -- in that order and in
the same order every time, whichever child skill produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ACK_ENV = "RELEASE_DOCS_ACK"


@dataclass(frozen=True)
class Finding:
    subject: str
    summary: str
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    evidence_total: int = 0


def render(findings: list[Finding], title: str) -> str:
    if not findings:
        return f"{title}: nothing stale.\n"

    lines = [f"{title}: {len(findings)} document(s) need attention.", ""]
    for finding in findings:
        lines.append(f"  {finding.subject}")
        lines.append(f"    {finding.summary}")
        if finding.detail:
            lines.append(f"    {finding.detail}")
        for path in finding.evidence:
            lines.append(f"      · {path}")
        hidden = finding.evidence_total - len(finding.evidence)
        if hidden > 0:
            lines.append(f"      · ... and {hidden} more")
        lines.append("")
    return "\n".join(lines) + "\n"


def acknowledgement() -> str | None:
    """An explicit, recorded reason to release with a document left alone.

    Deliberately a reason rather than a boolean. `RELEASE_DOCS_ACK=1` tells a
    later reader nothing, and the whole point of the gate is that silence about
    a document stops being possible.
    """
    value = (os.environ.get(ACK_ENV) or "").strip()
    if not value or value in {"1", "true", "yes"}:
        return None
    return value


def resolve(findings: list[Finding], title: str) -> tuple[str, int]:
    """Render findings and decide the exit status."""
    text = render(findings, title)
    if not findings:
        return text, 0

    ack = acknowledgement()
    if ack:
        return (
            text
            + f"Released anyway. {ACK_ENV}={ack!r}\n"
            "This reason belongs in the release notes.\n",
            0,
        )
    return (
        text
        + "Update these before tagging. They are the last step of a release,\n"
        "not the first step of the next one.\n\n"
        f'To release regardless, set {ACK_ENV} to a reason:\n'
        f'  {ACK_ENV}="user guide covers no changed surface" make release\n',
        1,
    )
