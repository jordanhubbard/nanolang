#!/usr/bin/env python3
"""
extract-docs.py — Auto-generate Markdown docs from /// doc comments in stdlib_*.c
and eval_*.c source files.

Doc comment format supported:
  /// Short description line (first line is used as the function signature)
  /// Longer description...
  ///
  /// @param name  Description
  /// @returns     Description
  ///
  /// @example
  ///   (some-call arg)   # => result

Also supports /** ... */ style block comments immediately before a function.

Usage:
  python3 tools/extract-docs.py [--src-dir src] [--out-dir docs/stdlib]

Output:
  One Markdown file per source module, e.g.:
    docs/stdlib/math.md
    docs/stdlib/string.md
    docs/stdlib/io.md
    docs/stdlib/runtime.md
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Source file → module name mapping
# ---------------------------------------------------------------------------
MODULE_MAP = {
    "eval_math.c":     "math",
    "eval_string.c":   "string",
    "eval_io.c":       "io",
    "eval_hashmap.c":  "hashmap",
    "stdlib_runtime.c": "runtime",
}

MODULE_TITLES = {
    "math":     "Math",
    "string":   "String",
    "io":       "I/O",
    "hashmap":  "Hashmap",
    "runtime":  "Runtime",
}

MODULE_DESCRIPTIONS = {
    "math":     "Built-in mathematical functions available without any imports.",
    "string":   "Built-in string manipulation functions.",
    "io":       "File system and OS interaction functions.",
    "hashmap":  "Key/value map operations.",
    "runtime":  "Low-level runtime helpers generated into transpiled C output.",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def strip_triple_slash(line: str) -> str:
    """Remove leading '/// ' or '///' from a line."""
    line = line.rstrip()
    if line.startswith("/// "):
        return line[4:]
    if line.startswith("///"):
        return line[3:]
    return line


def parse_triple_slash_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect consecutive /// lines starting at `start`. Returns (comment_lines, end_index)."""
    result = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("///"):
            result.append(strip_triple_slash(stripped))
            i += 1
        else:
            break
    return result, i


def parse_block_comment(lines: list[str], start: int) -> tuple[Optional[list[str]], int]:
    """
    Parse a /** ... */ block comment starting at `start`.
    Returns (comment_lines, end_index) or (None, start) if not a block comment.
    """
    if "/**" not in lines[start]:
        return None, start
    result = []
    i = start
    # Single-line /** ... */
    m = re.match(r"\s*/\*\*(.*?)\*/", lines[i])
    if m:
        return [m.group(1).strip()], i + 1
    # Multi-line
    # skip opening /**
    first_line = lines[i].replace("/**", "").strip(" *").rstrip()
    if first_line:
        result.append(first_line)
    i += 1
    while i < len(lines):
        line = lines[i]
        if "*/" in line:
            last = line[:line.index("*/")].strip(" *")
            if last:
                result.append(last)
            i += 1
            break
        # strip leading " * " or " *"
        stripped = re.sub(r"^\s*\*\s?", "", line).rstrip()
        result.append(stripped)
        i += 1
    return result, i


def c_function_name(line: str) -> Optional[str]:
    """
    Try to extract a C function name from a line like:
      Value builtin_abs(Value *args) {
      static int some_fn(int x) {
    Returns the identifier after the return type, or None.
    """
    # Match: <type(s)> <name>(<params...>) {
    m = re.match(
        r"^\s*(?:static\s+)?(?:const\s+)?(?:\w+\s+)+(\w+)\s*\(", line
    )
    if m:
        name = m.group(1)
        # Skip obvious non-function things
        if name in {"if", "for", "while", "switch", "return", "else"}:
            return None
        return name
    return None


def parse_doc_entry(comment_lines: list[str]) -> dict:
    """
    Parse structured fields from a list of comment lines.

    Expected structure:
      Line 0:  signature  e.g. "abs(x: int | float) -> int | float"
      Line 1:  short description
      ...
      @param name  Description
      @returns Description
      @example
        (code...)
    """
    entry: dict = {
        "signature": "",
        "description": [],
        "params": [],
        "returns": "",
        "examples": [],
    }

    if not comment_lines:
        return entry

    # First non-empty line that looks like a signature (has parens or ->)
    sig_idx = 0
    for idx, line in enumerate(comment_lines):
        line_s = line.strip()
        if line_s and ("(" in line_s or "->" in line_s):
            entry["signature"] = line_s
            sig_idx = idx + 1
            break
        elif line_s:
            # No signature line; treat as description
            sig_idx = 0
            break

    in_example = False
    for line in comment_lines[sig_idx:]:
        stripped = line.strip()
        if stripped.startswith("@param"):
            in_example = False
            m = re.match(r"@param\s+(\S+)\s+(.*)", stripped)
            if m:
                entry["params"].append((m.group(1), m.group(2)))
        elif stripped.startswith("@returns") or stripped.startswith("@return"):
            in_example = False
            m = re.match(r"@returns?\s+(.*)", stripped)
            if m:
                entry["returns"] = m.group(1)
        elif stripped == "@example":
            in_example = True
        elif in_example:
            entry["examples"].append(line)
        elif stripped:
            entry["description"].append(stripped)

    return entry


def scan_file(path: Path) -> list[dict]:
    """
    Scan a C file for /// or /** doc comment blocks immediately preceding
    a function definition. Returns a list of function doc dicts.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Warning: cannot read {path}: {e}", file=sys.stderr)
        return []

    lines = text.splitlines()
    results = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        comment_lines = None

        if stripped.startswith("///"):
            comment_lines, i = parse_triple_slash_block(lines, i)
        elif stripped.startswith("/**"):
            comment_lines, i = parse_block_comment(lines, i)
        else:
            i += 1
            continue

        if not comment_lines:
            i += 1
            continue

        # Skip blank lines between comment and function
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines):
            break

        fn_name = c_function_name(lines[i])
        if fn_name:
            entry = parse_doc_entry(comment_lines)
            entry["c_function"] = fn_name
            # Derive nano name: builtin_abs -> abs, builtin_file_read -> file_read
            nano_name = re.sub(r"^builtin_", "", fn_name)
            entry["name"] = nano_name
            if not entry["signature"]:
                entry["signature"] = nano_name + "(...)"
            results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def entry_to_markdown(entry: dict) -> str:
    lines = []
    sig = entry.get("signature", entry["name"])
    name = entry["name"]
    # Use name as the anchor id so TOC links work reliably
    lines.append(f"### `{sig}` {{ #{name} }}")
    lines.append("")

    desc = entry.get("description", [])
    if desc:
        lines.append(" ".join(desc))
        lines.append("")

    params = entry.get("params", [])
    if params:
        lines.append("**Parameters:**")
        lines.append("")
        for name, desc_p in params:
            lines.append(f"- `{name}` — {desc_p}")
        lines.append("")

    returns = entry.get("returns", "")
    if returns:
        lines.append(f"**Returns:** {returns}")
        lines.append("")

    examples = entry.get("examples", [])
    if examples:
        lines.append("**Example:**")
        lines.append("")
        lines.append("```nano")
        for ex in examples:
            # Dedent by 2 spaces if all lines are indented
            lines.append(ex[2:] if ex.startswith("  ") else ex)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def module_to_markdown(module: str, entries: list[dict]) -> str:
    title = MODULE_TITLES.get(module, module.capitalize())
    desc = MODULE_DESCRIPTIONS.get(module, "")

    lines = [
        f"# {title} Standard Library",
        "",
    ]
    if desc:
        lines += [desc, ""]

    lines += [
        f"> Auto-generated from source. Do not edit directly.",
        "",
        "---",
        "",
    ]

    if not entries:
        lines.append("*No documented functions found in this module.*")
        lines.append("")
        return "\n".join(lines)

    # Table of contents — use name only to keep anchors simple
    lines.append("## Functions")
    lines.append("")
    for entry in entries:
        name = entry["name"]
        sig = entry.get("signature", name)
        # anchor matches the heading "### `sig`" — mkdocs lowercases and replaces spaces with -
        # We link to the name anchor which is simpler and reliable
        lines.append(f"- [`{sig}`](#{name})")
    lines.append("")
    lines.append("---")
    lines.append("")

    for entry in entries:
        lines.append(entry_to_markdown(entry))
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract /// doc comments from nanolang stdlib C sources and generate Markdown."
    )
    parser.add_argument(
        "--src-dir",
        default="src",
        help="Root source directory to scan (default: src)",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/stdlib",
        help="Output directory for generated Markdown files (default: docs/stdlib)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information",
    )
    args = parser.parse_args()

    src_root = Path(args.src_dir)
    out_root = Path(args.out_dir)

    if not src_root.exists():
        print(f"Error: source directory '{src_root}' not found.", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    # Collect all stdlib_*.c and eval_*.c files
    source_files = list(src_root.rglob("stdlib_*.c")) + list(src_root.rglob("eval_*.c"))
    source_files.sort()

    if args.verbose:
        print(f"Scanning {len(source_files)} source file(s) in '{src_root}'...")

    module_entries: dict[str, list[dict]] = defaultdict(list)

    for path in source_files:
        module = MODULE_MAP.get(path.name)
        if module is None:
            # Auto-derive module name from filename
            name = path.stem
            if name.startswith("stdlib_"):
                module = name[len("stdlib_"):]
            elif name.startswith("eval_"):
                module = name[len("eval_"):]
            else:
                module = name

        entries = scan_file(path)
        if args.verbose:
            print(f"  {path.name}: {len(entries)} documented function(s) → module '{module}'")
        module_entries[module].extend(entries)

    # Write one Markdown file per module
    generated = []
    for module, entries in sorted(module_entries.items()):
        if not entries:
            continue
        md = module_to_markdown(module, entries)
        out_path = out_root / f"{module}.md"
        out_path.write_text(md, encoding="utf-8")
        generated.append(out_path)
        if args.verbose:
            print(f"  Wrote {out_path} ({len(entries)} function(s))")

    # Write an index page
    index_lines = [
        "# Standard Library Reference",
        "",
        "This reference is auto-generated from doc comments in the nanolang C source.",
        "",
        "## Modules",
        "",
    ]
    for module in sorted(module_entries.keys()):
        entries = module_entries[module]
        if not entries:
            continue
        title = MODULE_TITLES.get(module, module.capitalize())
        desc = MODULE_DESCRIPTIONS.get(module, "")
        index_lines.append(f"### [{title}]({module}.md)")
        if desc:
            index_lines.append(f"{desc}")
        index_lines.append(f"*{len(entries)} function(s) documented.*")
        index_lines.append("")

    index_path = out_root / "index.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    generated.append(index_path)

    print(f"Generated {len(generated)} file(s) in '{out_root}':")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
