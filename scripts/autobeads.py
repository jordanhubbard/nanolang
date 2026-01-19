#!/usr/bin/env python3
"""
autobeads.py

Run NanoLang tests/examples and automatically create/update Beads issues on failure.

Design goals:
- Low-friction: single command wrapper around existing Makefile targets
- Dedupe: avoid creating duplicate issues for the same underlying failure
- Actionable: include log tails + a stable fingerprint in the issue

This intentionally uses the Beads CLI (NOT markdown files).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _resolve_bd() -> str:
    # Prefer explicit override, then PATH, then legacy location.
    override = os.environ.get("BD")
    if override:
        return os.path.expanduser(override)
    which = shutil.which("bd")
    if which:
        return which
    return os.path.expanduser("~/.local/bin/bd")


BD = _resolve_bd()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_OUTPUT_DIR = PROJECT_ROOT / ".test_output"


@dataclass(frozen=True)
class Failure:
    kind: str  # "test_compile" | "test_runtime" | "examples"
    name: str
    log_paths: tuple[Path, ...]
    fingerprint: str
    summary: str


def _run(cmd: list[str], *, cwd: Path, timeout_s: int | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
        env=env,
    )


def _read_tail(path: Path, max_chars: int = 4000) -> str:
    try:
        data = path.read_text(errors="replace")
    except FileNotFoundError:
        return ""
    if len(data) <= max_chars:
        return data
    return data[-max_chars:]


def _fingerprint_from_text(text: str) -> str:
    # Keep it stable across line numbers/paths and small timing variations.
    # Strip absolute paths and line/col numbers.
    normalized = re.sub(r"/Users/[^\\s]+", "<ABS_PATH>", text)
    normalized = re.sub(r"line \\d+", "line <N>", normalized)
    normalized = re.sub(r"column \\d+", "column <N>", normalized)
    normalized = re.sub(r"0x[0-9a-fA-F]+", "0x<ADDR>", normalized)
    normalized = normalized.strip()
    h = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
    return h[:12]


def _ensure_test_output_dir() -> None:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_bd_issues() -> list[dict[str, Any]]:
    proc = _run([BD, "list", "--json"], cwd=PROJECT_ROOT, timeout_s=30)
    if proc.returncode != 0:
        raise RuntimeError(f"bd list failed (exit={proc.returncode}):\n{proc.stdout}")
    try:
        return json.loads(proc.stdout)
    except Exception as e:
        raise RuntimeError(f"Failed parsing bd list --json output: {e}\nRaw:\n{proc.stdout[:2000]}")

def _find_existing_issue_by_title(issues: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
    for issue in issues:
        if issue.get("status") != "open":
            continue
        if (issue.get("title") or "") == title:
            return issue
    return None


def _find_existing_issue_by_fingerprint(issues: list[dict[str, Any]], fingerprint: str) -> dict[str, Any] | None:
    needle = f"Fingerprint: {fingerprint}"
    for issue in issues:
        if issue.get("status") != "open":
            continue
        desc = issue.get("description") or ""
        notes = issue.get("notes") or ""
        if needle in desc or needle in notes:
            return issue
    return None


def _bd_create(title: str, description: str, priority: int, issue_type: str, labels: str, dry_run: bool) -> str | None:
    cmd = [
        BD,
        "create",
        "--silent",
        "--title",
        title,
        "--description",
        description,
        "--priority",
        str(priority),
        "--type",
        issue_type,
        "--labels",
        labels,
    ]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return None
    proc = _run(cmd, cwd=PROJECT_ROOT, timeout_s=30)
    if proc.returncode != 0:
        raise RuntimeError(f"bd create failed (exit={proc.returncode}):\n{proc.stdout}")
    issue_id = (proc.stdout or "").strip()
    if issue_id:
        return issue_id

    # Fallback: if output parsing fails, lookup by title.
    issues = _load_bd_issues()
    existing = _find_existing_issue_by_title(issues, title)
    return existing["id"] if existing else None


def _bd_add_note(issue_id: str, note: str, dry_run: bool) -> None:
    cmd = [BD, "update", issue_id, "--notes", note]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return
    proc = _run(cmd, cwd=PROJECT_ROOT, timeout_s=30)
    if proc.returncode != 0:
        raise RuntimeError(f"bd update --notes failed (exit={proc.returncode}):\n{proc.stdout}")

def _bd_close(issue_id: str, reason: str, dry_run: bool) -> None:
    cmd = [BD, "close", issue_id, "--reason", reason]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return
    proc = _run(cmd, cwd=PROJECT_ROOT, timeout_s=30)
    if proc.returncode != 0:
        raise RuntimeError(f"bd close failed (exit={proc.returncode}):\n{proc.stdout}")


def _collect_test_failures() -> list[Failure]:
    failures: list[Failure] = []

    # run_all_tests.sh leaves logs for failures; passes are deleted.
    compile_logs = sorted(TEST_OUTPUT_DIR.glob("*.compile.log"))
    for log in compile_logs:
        tail = _read_tail(log, max_chars=6000)
        if not tail.strip():
            # Sometimes the compiler crashes before writing; keep a placeholder.
            tail = "(empty compile log)"
        fp = _fingerprint_from_text(tail)
        failures.append(
            Failure(
                kind="test_compile",
                name=log.name.replace(".compile.log", ""),
                log_paths=(log,),
                fingerprint=fp,
                summary="Compilation failed",
            )
        )

    run_logs = sorted(TEST_OUTPUT_DIR.glob("*.run.log"))
    for log in run_logs:
        # If a run.log exists, it implies compilation succeeded but runtime failed.
        tail = _read_tail(log, max_chars=6000)
        if not tail.strip():
            tail = "(empty run log)"
        fp = _fingerprint_from_text(tail)
        failures.append(
            Failure(
                kind="test_runtime",
                name=log.name.replace(".run.log", ""),
                log_paths=(log,),
                fingerprint=fp,
                summary="Runtime (shadow test) failed",
            )
        )

    return failures


def _create_or_update_failures(failures: list[Failure], *, dry_run: bool, max_new: int) -> None:
    if not failures:
        return

    issues = _load_bd_issues()
    new_count = 0
    now = _dt.datetime.now().isoformat(timespec="seconds")

    for f in failures:
        # Dedupe strategy (per-failure mode):
        # 1) Prefer stable title match (kind+name). Fingerprints can legitimately change as logs evolve.
        # 2) Fall back to fingerprint match for legacy beads that used fingerprints only.
        title = f"[autotest] {f.summary}: {f.name}"
        existing = _find_existing_issue_by_title(issues, title) or _find_existing_issue_by_fingerprint(issues, f.fingerprint)
        logs_block = ""
        for p in f.log_paths:
            logs_block += f"\n---\nLog tail: {p}\n\n{_read_tail(p)}\n"

        if existing:
            note = (
                f"[autotest] Seen again at {now}\n"
                f"Kind: {f.kind}\n"
                f"Name: {f.name}\n"
                f"Fingerprint: {f.fingerprint}\n"
                f"{logs_block}"
            )
            _bd_add_note(existing["id"], note, dry_run=dry_run)
            continue

        if new_count >= max_new:
            # Don't spam; fold the rest into a single note on a summary bead (future enhancement).
            # For now, just print to stdout so CI logs show what was skipped.
            print(f"[autobeads] Max new issues reached ({max_new}); skipping: {f.kind} {f.name} fp={f.fingerprint}")
            continue

        desc = (
            f"Autogenerated by scripts/autobeads.py\n"
            f"\n"
            f"Kind: {f.kind}\n"
            f"Name: {f.name}\n"
            f"Key: {f.kind}:{f.name}\n"
            f"Fingerprint: {f.fingerprint}\n"
            f"First seen: {now}\n"
            f"{logs_block}"
        )
        labels = "autotest,test" if f.kind.startswith("test_") else "autotest,examples"
        _bd_create(title=title, description=desc, priority=4, issue_type="bug", labels=labels, dry_run=dry_run)
        new_count += 1


def _git(cmd: list[str]) -> str:
    proc = _run(["git"] + cmd, cwd=PROJECT_ROOT, timeout_s=10)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _current_branch() -> str:
    b = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    if b and b != "HEAD":
        return b
    sha = _git(["rev-parse", "--short", "HEAD"])
    return sha or "unknown"

def _detect_ci_job_name() -> str:
    # Prefer explicit CI job identifiers if present.
    candidates = [
        os.environ.get("CI_JOB_NAME"),          # GitLab CI
        os.environ.get("GITHUB_JOB"),           # GitHub Actions
        os.environ.get("GITHUB_WORKFLOW"),      # GitHub Actions (coarser)
        os.environ.get("BUILDKITE_LABEL"),      # Buildkite
        os.environ.get("BUILDKITE_JOB_ID"),     # Buildkite
        os.environ.get("CIRCLE_JOB"),           # CircleCI
        os.environ.get("TRAVIS_JOB_NAME"),      # Travis
        os.environ.get("JENKINS_JOB_NAME"),     # Jenkins (custom)
    ]
    for c in candidates:
        if c and c.strip():
            return c.strip()
    return "local"

def _sanitize_job_name(job: str) -> str:
    # Keep titles stable and readable.
    job = (job or "local").strip()
    job = re.sub(r"\s+", " ", job)
    # Avoid weird punctuation in bead titles
    job = re.sub(r"[^A-Za-z0-9 _./:-]+", "_", job)
    return job or "local"


def _summary_bead_title(kind: str, job_name: str) -> str:
    branch = _current_branch()
    job = _sanitize_job_name(job_name)
    return f"[autotest][summary] {kind} (branch={branch}, job={job})"


def _upsert_summary_bead(kind: str, job_name: str, *, dry_run: bool) -> str | None:
    title = _summary_bead_title(kind, job_name)
    issues = _load_bd_issues()
    existing = _find_existing_issue_by_title(issues, title)
    if existing:
        return existing["id"]

    now = _dt.datetime.now().isoformat(timespec="seconds")
    host = platform.node() or "unknown-host"
    sha = _git(["rev-parse", "--short", "HEAD"]) or "unknown"
    desc = (
        "Autogenerated by scripts/autobeads.py (summary mode)\n"
        "\n"
        f"Kind: {kind}\n"
        f"Branch: {_current_branch()}\n"
        f"Job: {_sanitize_job_name(job_name)}\n"
        f"Initial SHA: {sha}\n"
        f"Host: {host}\n"
        f"First seen: {now}\n"
    )
    labels = "autotest,summary"
    return _bd_create(title=title, description=desc, priority=3, issue_type="bug", labels=labels, dry_run=dry_run)


def _update_summary_bead(issue_id: str, kind: str, exit_code: int, failures: list[Failure], log_path: Path, *, dry_run: bool) -> None:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    sha = _git(["rev-parse", "--short", "HEAD"]) or "unknown"

    header = (
        f"[autotest][summary] Run at {now}\n"
        f"Kind: {kind}\n"
        f"SHA: {sha}\n"
        f"Exit: {exit_code}\n"
    )

    if failures:
        lines = [header, "\nFailures:\n"]
        for f in failures:
            lines.append(f"- {f.kind} {f.name} (fp={f.fingerprint})")
        lines.append("\n---\nmake log tail:\n\n" + _read_tail(log_path, max_chars=8000))
        for f in failures:
            for p in f.log_paths:
                lines.append(f"\n---\nLog tail: {p}\n\n{_read_tail(p, max_chars=6000)}\n")
        note = "\n".join(lines)
    else:
        note = header + "\n✅ No failures detected.\n"

    _bd_add_note(issue_id, note, dry_run=dry_run)


def _run_tests(timeout_s: int, *, dry_run: bool) -> int:
    _ensure_test_output_dir()
    # Always use makefile targets per repo rule.
    # NOTE: run test-impl to avoid recursion when `make test` itself calls autobeads.
    cmd = ["make", "test-impl"]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return 0
    proc = _run(cmd, cwd=PROJECT_ROOT, timeout_s=timeout_s)
    (TEST_OUTPUT_DIR / "make_test.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc.returncode


def _run_examples(timeout_s: int, *, dry_run: bool) -> int:
    _ensure_test_output_dir()
    cmd = ["make", "examples"]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return 0
    env = dict(os.environ)
    env["NANOLANG_AUTOBEADS_EXAMPLES"] = "1"
    proc = _run(cmd, cwd=PROJECT_ROOT, timeout_s=timeout_s, env=env)
    (TEST_OUTPUT_DIR / "make_examples.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc.returncode


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true", help="Run `make test` and create/update beads on failure")
    ap.add_argument("--examples", action="store_true", help="Run `make examples` and create/update beads on failure")
    ap.add_argument("--mode", choices=["per", "summary"], default="per", help="Beads mode: per-failure (local) or single summary bead (CI)")
    ap.add_argument("--close-on-success", action="store_true", help="In summary mode: close the summary bead when the run succeeds")
    ap.add_argument("--job-name", default=_detect_ci_job_name(), help="Job name used to key summary beads (default: auto-detect, falls back to 'local')")
    ap.add_argument("--timeout-seconds", type=int, default=1800, help="Timeout for make commands (default: 1800)")
    ap.add_argument("--dry-run", action="store_true", help="Print bd commands instead of executing them")
    ap.add_argument("--max-new", type=int, default=10, help="Max new beads created per run (default: 10)")
    args = ap.parse_args(argv)

    if not args.tests and not args.examples:
        ap.error("Must specify at least one of --tests or --examples")

    beads_enabled = Path(BD).exists()
    if not beads_enabled:
        # `make test` runs this script by default, but CI runners don't have the Beads
        # CLI installed. We still want CI to run the full test suite even when we
        # can't file issues.
        print(f"Warning: Beads CLI not found at {BD}; running without filing beads.", file=sys.stderr)

    exit_code = 0

    if args.tests:
        code = _run_tests(args.timeout_seconds, dry_run=args.dry_run)
        if beads_enabled:
            if args.mode == "per":
                if code != 0 and not args.dry_run:
                    failures = _collect_test_failures()
                    _create_or_update_failures(failures, dry_run=args.dry_run, max_new=args.max_new)
            else:
                issue_id = _upsert_summary_bead("make test", args.job_name, dry_run=args.dry_run)
                failures = _collect_test_failures() if (code != 0 and not args.dry_run) else []
                log_path = TEST_OUTPUT_DIR / "make_test.log"
                if issue_id:
                    _update_summary_bead(issue_id, "make test", code, failures, log_path, dry_run=args.dry_run)
                    if args.close_on_success and code == 0 and not args.dry_run:
                        _bd_close(issue_id, "CI run is green; auto-closing summary bead.", dry_run=args.dry_run)
        exit_code = exit_code or code

    if args.examples:
        code = _run_examples(args.timeout_seconds, dry_run=args.dry_run)
        log = TEST_OUTPUT_DIR / "make_examples.log"
        if beads_enabled:
            if args.mode == "per":
                if code != 0 and not args.dry_run:
                    # For examples, if make fails we currently create one bead with the log tail.
                    tail = _read_tail(log, max_chars=8000)
                    fp = _fingerprint_from_text(tail or "make examples failed")
                    failures = [
                        Failure(kind="examples", name="make examples", log_paths=(log,), fingerprint=fp, summary="Examples build failed")
                    ]
                    _create_or_update_failures(failures, dry_run=args.dry_run, max_new=args.max_new)
            else:
                issue_id = _upsert_summary_bead("make examples", args.job_name, dry_run=args.dry_run)
                failures: list[Failure] = []
                if code != 0 and not args.dry_run:
                    tail = _read_tail(log, max_chars=8000)
                    fp = _fingerprint_from_text(tail or "make examples failed")
                    failures = [Failure(kind="examples", name="make examples", log_paths=(log,), fingerprint=fp, summary="Examples build failed")]
                if issue_id:
                    _update_summary_bead(issue_id, "make examples", code, failures, log, dry_run=args.dry_run)
                    if args.close_on_success and code == 0 and not args.dry_run:
                        _bd_close(issue_id, "CI run is green; auto-closing summary bead.", dry_run=args.dry_run)
        exit_code = exit_code or code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

