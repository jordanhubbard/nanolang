#!/usr/bin/env python3

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Snippet:
    name: str
    source_path: Path
    code: str
    check: bool
    expect_stdout: str | None
    compile_timeout_s: float
    run_timeout_s: float


SNIPPET_MARKER_RE = re.compile(r"<!--\s*nl-snippet\s*(\{.*\})\s*-->")
FENCE_START_RE = re.compile(r"^```\s*(\w+)?\s*$")
FENCE_END_RE = re.compile(r"^```\s*$")


def repo_root() -> Path:
    # scripts/ is at repo_root/scripts/
    return Path(__file__).resolve().parents[1]


def iter_markdown_files(root: Path) -> list[Path]:
    base = root / "userguide"
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.md") if p.is_file())


def parse_snippets(md_path: Path) -> list[Snippet]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    out: list[Snippet] = []

    i = 0
    while i < len(lines):
        m = SNIPPET_MARKER_RE.search(lines[i])
        if not m:
            i += 1
            continue

        try:
            meta = json.loads(m.group(1))
        except Exception as exc:
            raise RuntimeError(f"Invalid nl-snippet JSON in {md_path}:{i+1}: {exc}") from exc

        name = str(meta.get("name") or "").strip()
        if not name:
            raise RuntimeError(f"Missing snippet name in {md_path}:{i+1}")

        check = bool(meta.get("check", True))
        expect_stdout = meta.get("expect_stdout")
        if expect_stdout is not None:
            expect_stdout = str(expect_stdout)
        compile_timeout_s = float(meta.get("compile_timeout_s", 30))
        run_timeout_s = float(meta.get("run_timeout_s", 5))

        # Find the next fenced code block.
        j = i + 1
        while j < len(lines) and not FENCE_START_RE.match(lines[j]):
            j += 1
        if j >= len(lines):
            raise RuntimeError(f"Snippet '{name}' in {md_path}:{i+1} missing fenced code block")

        fence_lang = (FENCE_START_RE.match(lines[j]).group(1) or "").strip().lower()
        if fence_lang not in {"nano", "nanolang"}:
            raise RuntimeError(
                f"Snippet '{name}' in {md_path}:{j+1} must use ```nano (found '{fence_lang}')"
            )

        j += 1
        code_lines: list[str] = []
        while j < len(lines) and not FENCE_END_RE.match(lines[j]):
            code_lines.append(lines[j])
            j += 1
        if j >= len(lines):
            raise RuntimeError(f"Snippet '{name}' in {md_path}:{i+1} missing closing ```")

        code = "\n".join(code_lines).rstrip() + "\n"
        out.append(
            Snippet(
                name=name,
                source_path=md_path,
                code=code,
                check=check,
                expect_stdout=expect_stdout,
                compile_timeout_s=compile_timeout_s,
                run_timeout_s=run_timeout_s,
            )
        )

        i = j + 1

    return out


def run(cmd: list[str], *, cwd: Path, timeout_s: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def main() -> int:
    root = repo_root()
    md_files = iter_markdown_files(root)
    if not md_files:
        print("userguide/: no markdown files found", file=sys.stderr)
        return 1

    snippets: list[Snippet] = []
    for md in md_files:
        snippets.extend(parse_snippets(md))

    runnable = [s for s in snippets if s.check]
    if not runnable:
        print("userguide/: no runnable snippets found (all snippets have check=false)", file=sys.stderr)
        return 1

    out_dir = root / "build" / "userguide" / "snippets"
    bin_dir = out_dir / "bin"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)

    nanoc = root / "bin" / "nanoc"
    if not nanoc.exists():
        print("Missing compiler at bin/nanoc; run `make build` first", file=sys.stderr)
        return 1

    failures: list[str] = []
    for snip in runnable:
        src = out_dir / f"{snip.name}.nano"
        exe = bin_dir / snip.name
        src.write_text(snip.code, encoding="utf-8")

        compile_res = run(
            [str(nanoc), str(src), "-o", str(exe)],
            cwd=root,
            timeout_s=snip.compile_timeout_s,
        )
        if compile_res.returncode != 0:
            failures.append(
                "\n".join(
                    [
                        f"[FAIL] compile: {snip.name} ({snip.source_path})",
                        compile_res.stdout,
                        compile_res.stderr,
                    ]
                )
            )
            continue

        run_res = run([str(exe)], cwd=root, timeout_s=snip.run_timeout_s)
        if run_res.returncode != 0:
            failures.append(
                "\n".join(
                    [
                        f"[FAIL] run: {snip.name} ({snip.source_path})",
                        run_res.stdout,
                        run_res.stderr,
                    ]
                )
            )
            continue

        if snip.expect_stdout is not None and run_res.stdout != snip.expect_stdout:
            failures.append(
                "\n".join(
                    [
                        f"[FAIL] stdout mismatch: {snip.name} ({snip.source_path})",
                        "--- expected ---",
                        snip.expect_stdout,
                        "--- got ---",
                        run_res.stdout,
                    ]
                )
            )

    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        print(f"\nuserguide-check: {len(failures)} snippet(s) failed", file=sys.stderr)
        return 1

    print(f"userguide-check: {len(runnable)} snippet(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
