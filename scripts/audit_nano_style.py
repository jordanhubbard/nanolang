#!/usr/bin/env python3
"""Report conservative, lexical NanoLang repository style findings.

I do not rewrite source. Report mode always succeeds; strict mode fails for
findings not recorded in the baseline. Use --no-baseline to make every finding
strict, or --write-baseline to deliberately accept the current inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


METADATA_FIELDS = (
    "Example",
    "Purpose",
    "Features",
    "Difficulty",
    "Category",
    "Prerequisites",
    "Track",
    "Build",
    "Dependencies",
    "Tags",
    "Expected Output",
)
RETIRED_ROOT_STDLIB = {"async", "iter", "list", "map", "option", "result", "set", "string"}
IMPORT_RE = re.compile(
    r'^\s*(?:from\s+|import\s+|(?:unsafe\s+)?module\s+)["\']([^"\']+\.nano)["\']'
)
FUNCTION_RE = re.compile(r"^\s*(?P<extern>extern\s+)?(?:pub\s+)?fn\s+(?P<name>[A-Za-z_]\w*)")
SHADOW_RE = re.compile(r"^\s*shadow\s+(?P<name>[A-Za-z_]\w*)")


@dataclass(frozen=True)
class Finding:
    category: str
    path: str
    line: int
    detail: str

    @property
    def fingerprint(self) -> str:
        # Deliberately omit the line number so unrelated edits do not invalidate debt.
        raw = f"{self.category}\0{self.path}\0{self.detail}".encode()
        return hashlib.sha256(raw).hexdigest()[:20]


def mask_line(line: str, in_block_comment: bool) -> tuple[str, bool, str | None]:
    """Mask strings/comments while retaining columns; return bad comment syntax."""
    out = list(line)
    i = 0
    unsupported = None
    while i < len(line):
        if in_block_comment:
            end = line.find("*/", i)
            if end < 0:
                return "".join(" " for _ in line), True, None
            for j in range(i, end + 2):
                out[j] = " "
            i = end + 2
            in_block_comment = False
            continue
        if line.startswith("/*", i):
            out[i] = out[i + 1] = " "
            in_block_comment = True
            i += 2
            continue
        if line[i] in {'"', "'"}:
            quote = line[i]
            out[i] = " "
            i += 1
            while i < len(line):
                char = line[i]
                out[i] = " "
                i += 1
                if char == "\\" and i < len(line):
                    out[i] = " "
                    i += 1
                elif char == quote:
                    break
            continue
        if line[i] == "#":
            for j in range(i, len(line)):
                out[j] = " "
            break
        if line.startswith("//", i) or line.startswith("--", i):
            unsupported = line[i : i + 2]
            for j in range(i, len(line)):
                out[j] = " "
            break
        i += 1
    return "".join(out), in_block_comment, unsupported


def exemption_for_example(path: Path, lines: list[str]) -> str | None:
    rel = path.as_posix()
    header = "\n".join(lines[:30]).lower()
    if rel.startswith("examples/lib/") or rel.startswith("examples/large_project/src/"):
        return "internal-support"
    if "# track: internal" in header:
        return "internal-track"
    if "harness" in path.stem.lower() or "fixture" in path.stem.lower():
        return "harness-or-fixture"
    if "/output/" in f"/{rel}" or "# build: generated" in header:
        return "generated"
    return None


def resolve_import(root: Path, source: Path, imported: str) -> bool:
    candidates = [root / imported, source.parent / imported]
    if imported.startswith("std/"):
        candidates.append(root / "modules" / imported)
    return any(candidate.is_file() for candidate in candidates)


def audit_file(root: Path, path: Path) -> tuple[list[Finding], Counter[str]]:
    rel = path.relative_to(root).as_posix()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [Finding("encoding", rel, 1, "file is not valid UTF-8")], Counter()

    findings: list[Finding] = []
    exemptions: Counter[str] = Counter()
    rel_path = Path(rel)
    if (
        len(rel_path.parts) == 2
        and rel_path.parts[0] == "stdlib"
        and rel_path.stem in RETIRED_ROOT_STDLIB
    ):
        findings.append(Finding("root-stdlib-dialect", rel, 1, "retired root stdlib module remains"))
    masked: list[str] = []
    in_block = False
    functions: list[tuple[str, int, bool]] = []
    shadows: set[str] = set()

    for number, line in enumerate(lines, 1):
        code, in_block, unsupported = mask_line(line, in_block)
        masked.append(code)
        if unsupported:
            findings.append(Finding("unsupported-comment", rel, number, f"use # or /* */ instead of {unsupported}"))

        match = FUNCTION_RE.match(code)
        if match:
            functions.append((match.group("name"), number, bool(match.group("extern"))))
        shadow = SHADOW_RE.match(code)
        if shadow:
            shadows.add(shadow.group("name"))

        imported = IMPORT_RE.match(line)
        if imported:
            imported_path = imported.group(1)
            parts = Path(imported_path).parts
            if len(parts) == 2 and parts[0] == "stdlib" and Path(parts[1]).stem in RETIRED_ROOT_STDLIB:
                findings.append(Finding("root-stdlib-dialect", rel, number, f"imports retired {imported_path}"))
            if not resolve_import(root, path, imported_path):
                findings.append(Finding("stale-path", rel, number, f"import target does not exist: {imported_path}"))

    code_text = "\n".join(masked)
    for match in re.finditer(r"\bassert\s+(true|\(\s*==\s+1\s+1\s*\))", code_text):
        line = code_text.count("\n", 0, match.start()) + 1
        expression = re.sub(r"\s+", " ", match.group(0))
        findings.append(Finding("vacuous-shadow", rel, line, expression))

    for name, number, is_extern in functions:
        if is_extern:
            exemptions["shadow:extern"] += 1
        elif name not in shadows:
            findings.append(Finding("missing-shadow", rel, number, f"function {name} has no matching shadow"))

    if rel.startswith("examples/"):
        exemption = exemption_for_example(Path(rel), lines)
        if exemption:
            exemptions[f"metadata:{exemption}"] += 1
        else:
            present = {
                match.group(1)
                for line in lines[:30]
                if (match := re.match(r"^# (.+?):\s*\S", line))
            }
            missing = [field for field in METADATA_FIELDS if field not in present]
            if missing:
                findings.append(Finding("example-metadata", rel, 1, "missing fields: " + ", ".join(missing)))

    return findings, exemptions


def scan(root: Path) -> tuple[list[Finding], Counter[str]]:
    findings: list[Finding] = []
    exemptions: Counter[str] = Counter()
    for path in sorted(root.rglob("*.nano")):
        if any(part in {".git", "build", "vendor", "node_modules", "_cache"} for part in path.parts):
            continue
        file_findings, file_exemptions = audit_file(root, path)
        findings.extend(file_findings)
        exemptions.update(file_exemptions)
    return sorted(findings, key=lambda item: (item.path, item.line, item.category, item.detail)), exemptions


def load_baseline(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("fingerprints", []))


def write_baseline(path: Path, findings: list[Finding]) -> None:
    data = {
        "version": 1,
        "description": "Accepted NanoLang style findings. Regenerate only after reviewing the report.",
        "fingerprints": sorted({finding.fingerprint for finding in findings}),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--strict", action="store_true", help="fail when findings are not in the baseline")
    parser.add_argument("--baseline", type=Path, help="baseline path (default: <root>/scripts/nano_style_baseline.json)")
    parser.add_argument("--no-baseline", action="store_true", help="treat every finding as new")
    parser.add_argument("--write-baseline", action="store_true", help="replace the baseline with the current inventory")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    baseline_path = args.baseline or root / "scripts" / "nano_style_baseline.json"
    findings, exemptions = scan(root)
    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        write_baseline(baseline_path, findings)

    accepted = set() if args.no_baseline else load_baseline(baseline_path)
    new_findings = [finding for finding in findings if finding.fingerprint not in accepted]
    for finding in findings:
        status = "NEW" if finding in new_findings else "BASELINE"
        print(f"{status} {finding.category} {finding.path}:{finding.line}: {finding.detail}")

    counts = Counter(finding.category for finding in findings)
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"
    print(f"Summary: {len(findings)} finding(s), {len(new_findings)} new; {summary}")
    if exemptions:
        print("Exemptions: " + ", ".join(f"{key}={exemptions[key]}" for key in sorted(exemptions)))
    if args.write_baseline:
        print(f"Baseline written: {baseline_path}")
    return 1 if args.strict and new_findings else 0


if __name__ == "__main__":
    sys.exit(main())
