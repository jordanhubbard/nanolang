#!/usr/bin/env python3
"""
build_docs.py — nanolang static documentation site generator
Converts userguide/**/*.md to a self-contained HTML site in site/

Usage:
    python3 scripts/build_docs.py [--output site] [--title "NanoLang Docs"]
    make docs         # runs this script + opens site/index.html
    make docs-serve   # runs a local dev server on port 8080

Features:
    - Zero external dependencies (Python 3.6+ stdlib only)
    - Syntax highlighting for ```nano code blocks (keyword-based)
    - Navigation sidebar with section tree
    - Single-page TOC per document
    - Responsive CSS (mobile-friendly)
    - Preserves <!-- --> HTML comments (for nl-snippet metadata)
    - make docs-serve spins up http.server for local preview
"""

import os
import re
import sys
import html
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="nanolang docs static site generator")
parser.add_argument("--output",     default="site",         help="Output directory")
parser.add_argument("--input",      default="userguide",    help="Input directory")
parser.add_argument("--title",      default="NanoLang",     help="Site title")
parser.add_argument("--version",    default="dev",          help="Version string")
parser.add_argument("--root",       default="",             help="Site root prefix (for GitHub Pages)")
parser.add_argument("--clean",      action="store_true",    help="Remove output dir before building")
args = parser.parse_args()

REPO_ROOT = Path(__file__).parent.parent
INPUT_DIR = REPO_ROOT / args.input
OUTPUT_DIR = REPO_ROOT / args.output
SITE_ROOT = args.root.rstrip("/")
TITLE = args.title
VERSION = args.version
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ── Nano syntax highlighting ──────────────────────────────────────────────────
NANO_KEYWORDS = {
    "fn", "return", "let", "set", "mut", "if", "else", "while", "for",
    "struct", "enum", "union", "shadow", "pub", "extern", "import", "from",
    "module", "assert", "true", "false", "and", "or", "not", "as",
    "match", "with", "effect", "handle", "raise", "async", "await",
    "gpu", "par",
}
NANO_TYPES = {
    "int", "float", "bool", "string", "void", "T", "U", "V",
    "List", "Maybe", "Result", "Option",
}

def highlight_nano(code: str) -> str:
    """Apply keyword-based syntax highlighting to a nano code block."""
    lines = []
    for line in code.split("\n"):
        # Escape HTML first
        escaped = html.escape(line)
        # Comments: /* */ style
        escaped = re.sub(
            r"(/\*.*?\*/)",
            r'<span class="c">\1</span>',
            escaped, flags=re.DOTALL
        )
        # Line comments //
        escaped = re.sub(
            r"(//[^\n]*)",
            r'<span class="c">\1</span>',
            escaped
        )
        # String literals
        escaped = re.sub(
            r'(&quot;[^&]*?&quot;)',
            r'<span class="s">\1</span>',
            escaped
        )
        # f-string prefix
        escaped = re.sub(
            r'\bf(&quot;)',
            r'<span class="kw">f</span><span class="s">\1</span>',
            escaped
        )
        # Numbers
        escaped = re.sub(
            r'\b(\d+(?:\.\d+)?)\b',
            r'<span class="n">\1</span>',
            escaped
        )
        # Keywords
        for kw in NANO_KEYWORDS:
            escaped = re.sub(
                rf'\b({kw})\b',
                r'<span class="kw">\1</span>',
                escaped
            )
        # Types
        for tp in NANO_TYPES:
            escaped = re.sub(
                rf'\b({tp})\b',
                r'<span class="tp">\1</span>',
                escaped
            )
        # Operators
        escaped = re.sub(
            r'(\+|-|\*|/|%|==|!=|&lt;=|&gt;=|&lt;|&gt;|-&gt;)',
            r'<span class="op">\1</span>',
            escaped
        )
        lines.append(escaped)
    return "\n".join(lines)

# ── Minimal Markdown → HTML converter ────────────────────────────────────────

def md_to_html(md: str, page_path: str = "") -> tuple[str, list]:
    """Convert Markdown to HTML. Returns (html_body, toc_entries)."""
    lines = md.split("\n")
    out = []
    toc = []
    i = 0
    slug_counts: dict = {}

    def make_slug(text: str) -> str:
        """Create a URL-safe anchor slug from heading text."""
        raw = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[\s]+", "-", raw.strip())
        count = slug_counts.get(slug, 0)
        slug_counts[slug] = count + 1
        return slug if count == 0 else f"{slug}-{count}"

    def inline(text: str) -> str:
        """Process inline markdown."""
        # Bold + italic
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
        text = re.sub(r"_(.+?)_",       r"<em>\1</em>",         text)
        # Inline code
        text = re.sub(r"`([^`]+)`",
                      lambda m: f"<code>{html.escape(m.group(1))}</code>", text)
        # Links
        text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
        # Auto-links
        text = re.sub(r"<(https?://[^>]+)>", r'<a href="\1">\1</a>', text)
        return text

    in_fence = False
    fence_lang = ""
    fence_lines: list = []
    in_list = False
    in_blockquote = False
    in_table = False
    table_rows: list = []

    def flush_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def flush_blockquote():
        nonlocal in_blockquote
        if in_blockquote:
            out.append("</blockquote>")
            in_blockquote = False

    def flush_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            out.append('<table class="data-table">')
            for ri, row in enumerate(table_rows):
                cells = [c.strip() for c in row.strip("|").split("|")]
                tag = "th" if ri == 0 else "td"
                if ri == 1 and all(re.match(r"[-: ]+$", c) for c in cells):
                    continue  # separator row
                out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            out.append("</table>")
            in_table = False
            table_rows = []

    while i < len(lines):
        line = lines[i]

        # ── Fenced code block ───────────────────────────────────────────
        if re.match(r"^```", line):
            if not in_fence:
                flush_list(); flush_blockquote(); flush_table()
                fence_lang = line[3:].strip().lower()
                # Strip nl-snippet metadata comments
                fence_lang = re.sub(r"\s*<!--.*?-->", "", fence_lang).strip()
                in_fence = True
                fence_lines = []
            else:
                code_raw = "\n".join(fence_lines)
                if fence_lang in ("nano", "nanolang"):
                    highlighted = highlight_nano(code_raw)
                    out.append(f'<pre class="code-block lang-nano"><code>{highlighted}</code></pre>')
                elif fence_lang:
                    out.append(f'<pre class="code-block lang-{html.escape(fence_lang)}"><code>{html.escape(code_raw)}</code></pre>')
                else:
                    out.append(f'<pre class="code-block"><code>{html.escape(code_raw)}</code></pre>')
                in_fence = False
            i += 1
            continue

        if in_fence:
            fence_lines.append(line)
            i += 1
            continue

        # ── HTML comments — pass through ────────────────────────────────
        if line.strip().startswith("<!--"):
            i += 1
            continue

        # ── Headings ────────────────────────────────────────────────────
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            flush_list(); flush_blockquote(); flush_table()
            level = len(m.group(1))
            text_raw = m.group(2).strip()
            text_html = inline(text_raw)
            slug = make_slug(re.sub(r"<[^>]+>", "", text_html))
            out.append(f'<h{level} id="{slug}">{text_html} <a class="anchor" href="#{slug}">¶</a></h{level}>')
            if level <= 3:
                toc.append((level, text_raw, slug))
            i += 1
            continue

        # ── Horizontal rule ─────────────────────────────────────────────
        if re.match(r"^---+$|^===+$|\*\*\*+$", line.strip()):
            flush_list(); flush_blockquote(); flush_table()
            out.append("<hr>")
            i += 1
            continue

        # ── Table ───────────────────────────────────────────────────────
        if "|" in line and not in_fence:
            flush_list(); flush_blockquote()
            in_table = True
            table_rows.append(line)
            i += 1
            continue
        elif in_table:
            flush_table()

        # ── Blockquote ──────────────────────────────────────────────────
        m = re.match(r"^>\s*(.*)", line)
        if m:
            flush_list()
            if not in_blockquote:
                out.append("<blockquote>")
                in_blockquote = True
            out.append(f"<p>{inline(m.group(1))}</p>")
            i += 1
            continue
        else:
            flush_blockquote()

        # ── Unordered list ──────────────────────────────────────────────
        m = re.match(r"^[-*+]\s+(.*)", line)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(m.group(1))}</li>")
            i += 1
            continue

        # ── Ordered list ────────────────────────────────────────────────
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            if not in_list:
                out.append('<ol>')
                in_list = True
            out.append(f"<li>{inline(m.group(1))}</li>")
            i += 1
            continue

        # ── Empty line ──────────────────────────────────────────────────
        if line.strip() == "":
            flush_list(); flush_table()
            out.append("")
            i += 1
            continue

        # ── Paragraph ───────────────────────────────────────────────────
        flush_list(); flush_blockquote(); flush_table()
        # Collect contiguous non-blank lines into one paragraph
        para_lines = []
        while i < len(lines) and lines[i].strip() != "" and not re.match(r"^[#`>|-]", lines[i]):
            para_lines.append(lines[i])
            i += 1
        para = " ".join(para_lines)
        if para.strip():
            out.append(f"<p>{inline(para)}</p>")
        continue

    flush_list(); flush_blockquote(); flush_table()
    if in_fence and fence_lines:
        # Unclosed fence
        out.append(f'<pre class="code-block"><code>{html.escape(chr(10).join(fence_lines))}</code></pre>')

    return "\n".join(out), toc

# ── HTML page template ────────────────────────────────────────────────────────

CSS = """
/* nano-docs — minimal, clean, readable */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --code-bg: #1e2734; --kw: #ff7b72; --tp: #79c0ff;
    --str: #a5d6ff; --num: #f2cc60; --op: #79c0ff; --cm: #6e7681;
    --nav-w: 260px;
}
@media (prefers-color-scheme: light) {
    :root {
        --bg: #fff; --surface: #f6f8fa; --border: #d0d7de;
        --text: #1f2328; --muted: #57606a; --accent: #0969da;
        --code-bg: #f6f8fa; --kw: #cf222e; --tp: #0550ae;
        --str: #0a3069; --num: #953800; --op: #0550ae; --cm: #6e7781;
    }
}
html { font-size: 16px; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    display: flex; min-height: 100vh;
}
/* Navigation */
nav#sidebar {
    width: var(--nav-w); min-width: var(--nav-w);
    background: var(--surface); border-right: 1px solid var(--border);
    position: sticky; top: 0; height: 100vh; overflow-y: auto;
    padding: 1.5rem 0; flex-shrink: 0;
}
nav#sidebar .logo { padding: 0 1.25rem 1rem; border-bottom: 1px solid var(--border); }
nav#sidebar .logo a { color: var(--text); text-decoration: none; font-size: 1.1rem; font-weight: 600; }
nav#sidebar .logo .ver { font-size: .75rem; color: var(--muted); display: block; margin-top: .15rem; }
nav#sidebar .section { padding: .75rem 1.25rem .25rem; font-size: .7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
nav#sidebar ul { list-style: none; }
nav#sidebar ul li a {
    display: block; padding: .3rem 1.25rem;
    color: var(--muted); text-decoration: none; font-size: .875rem;
    border-left: 2px solid transparent; transition: all .1s;
}
nav#sidebar ul li a:hover, nav#sidebar ul li a.active {
    color: var(--text); border-left-color: var(--accent);
    background: rgba(88,166,255,.06);
}
/* Main */
main { flex: 1; max-width: 860px; padding: 2rem 3rem; min-width: 0; }
h1, h2, h3, h4 { color: var(--text); margin: 1.5rem 0 .5rem; font-weight: 600; line-height: 1.3; }
h1 { font-size: 2rem; border-bottom: 1px solid var(--border); padding-bottom: .4rem; }
h2 { font-size: 1.4rem; border-bottom: 1px solid var(--border); padding-bottom: .3rem; }
h3 { font-size: 1.1rem; } h4 { font-size: 1rem; color: var(--muted); }
p  { margin: .6rem 0; color: var(--text); }
ul, ol { margin: .5rem 0 .5rem 1.5rem; }
li { margin: .2rem 0; }
a  { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
blockquote { border-left: 3px solid var(--accent); padding: .5rem 1rem;
    background: var(--surface); margin: .75rem 0; border-radius: 0 4px 4px 0; }
code { background: var(--code-bg); padding: .1em .35em; border-radius: 4px;
    font-size: .875em; font-family: 'Cascadia Code', 'Fira Code', monospace; color: var(--tp); }
pre.code-block { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem 1.25rem; margin: .75rem 0; overflow-x: auto; }
pre.code-block code { background: none; padding: 0; color: var(--text); font-size: .84rem;
    font-family: 'Cascadia Code', 'Fira Code', monospace; }
/* Syntax highlighting */
.kw { color: var(--kw); font-weight: 600; }
.tp { color: var(--tp); }
.s  { color: var(--str); }
.n  { color: var(--num); }
.op { color: var(--op); }
.c  { color: var(--cm); font-style: italic; }
.anchor { font-size: .7em; opacity: .4; margin-left: .4em; text-decoration: none; }
.anchor:hover { opacity: 1; }
table.data-table { border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .875rem; }
table.data-table th, table.data-table td { border: 1px solid var(--border); padding: .4rem .75rem; text-align: left; }
table.data-table th { background: var(--surface); font-weight: 600; }
.toc { background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem 1.25rem; margin: 1.5rem 0; font-size: .875rem; }
.toc ul { margin-left: 1rem; }
.toc a { color: var(--muted); }
.toc a:hover { color: var(--accent); }
footer { font-size: .75rem; color: var(--muted); margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid var(--border); }
@media (max-width: 768px) {
    body { flex-direction: column; }
    nav#sidebar { width: 100%; min-width: 0; height: auto; position: relative; }
    main { padding: 1rem 1.25rem; }
}
"""

def make_toc_html(toc: list) -> str:
    if not toc: return ""
    out = ['<nav class="toc"><strong>On this page</strong><ul>']
    for level, text, slug in toc:
        indent = "&nbsp;" * ((level - 1) * 2)
        out.append(f'<li>{indent}<a href="#{slug}">{html.escape(text)}</a></li>')
    out.append("</ul></nav>")
    return "\n".join(out)

def make_nav_html(nav_tree: list, current_path: str) -> str:
    """Build the sidebar nav HTML."""
    out = []
    current_section = None
    for entry in nav_tree:
        if entry["type"] == "section":
            if current_section:
                out.append("</ul>")
            current_section = entry["title"]
            out.append(f'<div class="section">{html.escape(entry["title"])}</div>')
            out.append("<ul>")
        elif entry["type"] == "page":
            path = entry["path"]
            active = "active" if path == current_path else ""
            out.append(f'<li><a href="{SITE_ROOT}/{path}" class="{active}">{html.escape(entry["title"])}</a></li>')
    if current_section:
        out.append("</ul>")
    return "\n".join(out)

def page_template(title: str, body: str, toc_html: str, nav_html: str, rel_path: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — {html.escape(TITLE)}</title>
<style>{CSS}</style>
</head>
<body>
<nav id="sidebar">
  <div class="logo">
    <a href="{SITE_ROOT}/index.html">🦀 {html.escape(TITLE)}</a>
    <span class="ver">v{html.escape(VERSION)}</span>
  </div>
  {nav_html}
</nav>
<main>
{toc_html}
{body}
<footer>Generated {NOW} · <a href="https://github.com/jordanhubbard/nanolang">GitHub</a></footer>
</main>
</body>
</html>"""

# ── Discover source files ─────────────────────────────────────────────────────

def collect_pages() -> list:
    """Walk INPUT_DIR and collect all .md files."""
    pages = []
    for md_path in sorted(INPUT_DIR.rglob("*.md")):
        rel = md_path.relative_to(INPUT_DIR)
        pages.append({
            "src":   md_path,
            "rel":   rel,
            "depth": len(rel.parts) - 1,
        })
    return pages

def page_title(md_path: Path) -> str:
    """Extract the first # heading from a markdown file."""
    try:
        content = md_path.read_text(encoding="utf-8")
        for line in content.split("\n"):
            m = re.match(r"^#\s+(.+)$", line)
            if m:
                return m.group(1).strip()
        return md_path.stem.replace("_", " ").title()
    except Exception:
        return md_path.stem

def build_nav_tree(pages: list) -> list:
    """Build navigation tree from pages list."""
    tree = []
    seen_sections = set()
    for p in pages:
        parts = p["rel"].parts
        section = parts[0].replace("_", " ").title() if len(parts) > 1 else "Reference"
        if section not in seen_sections:
            tree.append({"type": "section", "title": section})
            seen_sections.add(section)
        html_rel = str(p["rel"]).replace(".md", ".html")
        tree.append({
            "type":  "page",
            "title": page_title(p["src"]),
            "path":  html_rel,
        })
    return tree

# ── Build site ────────────────────────────────────────────────────────────────

def build():
    if args.clean and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pages = collect_pages()
    if not pages:
        print(f"No .md files found in {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    nav_tree = build_nav_tree(pages)

    print(f"Building {len(pages)} pages → {OUTPUT_DIR}/")

    for p in pages:
        src   = p["src"]
        rel   = p["rel"]
        dst   = OUTPUT_DIR / rel.with_suffix(".html")
        dst.parent.mkdir(parents=True, exist_ok=True)

        content = src.read_text(encoding="utf-8")
        body, toc = md_to_html(content, str(rel))

        html_rel = str(rel).replace(".md", ".html")
        nav_html = make_nav_html(nav_tree, html_rel)
        toc_html = make_toc_html(toc)
        title    = page_title(src)

        html_out = page_template(title, body, toc_html, nav_html, html_rel)
        dst.write_text(html_out, encoding="utf-8")
        print(f"  {rel} → {dst.relative_to(REPO_ROOT)}")

    # Generate index.html (redirect or content from README)
    readme = REPO_ROOT / "README.md"
    index_src = readme if readme.exists() else (INPUT_DIR / "index.md") if (INPUT_DIR / "index.md").exists() else None
    if index_src:
        content = index_src.read_text(encoding="utf-8")
        body, toc = md_to_html(content, "index.md")
    else:
        # Auto-generate index from nav
        links = "\n".join(
            f'<li><a href="{SITE_ROOT}/{e["path"]}">{html.escape(e["title"])}</a></li>'
            for e in nav_tree if e["type"] == "page"
        )
        body = f"<h1>{html.escape(TITLE)} Documentation</h1><ul>{links}</ul>"
        toc  = []

    nav_html = make_nav_html(nav_tree, "index.html")
    toc_html = make_toc_html(toc)
    idx_html = page_template(f"{TITLE} Docs", body, toc_html, nav_html, "index.html")
    (OUTPUT_DIR / "index.html").write_text(idx_html, encoding="utf-8")
    print(f"  index.html → {OUTPUT_DIR / 'index.html'}")

    # Copy any static assets from docs/ into site/
    docs_static = REPO_ROOT / "docs"
    if docs_static.is_dir():
        for f in docs_static.glob("*.png"):
            shutil.copy(f, OUTPUT_DIR / f.name)

    print(f"\n✅ Site built: {OUTPUT_DIR}/index.html  ({len(pages)+1} pages)")

if __name__ == "__main__":
    build()
