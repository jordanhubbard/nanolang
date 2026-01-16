#!/usr/bin/env python3

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from os import path as ospath


FENCE_START_RE = re.compile(r"^```\s*(\w+)?\s*$")
FENCE_END_RE = re.compile(r"^```\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
SNIPPET_MARKER_RE = re.compile(r"^\s*<!--\s*nl-snippet\b")

HIGHLIGHT_TOOL = Path("tools") / "nano_highlight.nano"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel_href(target: Path, from_dir: Path) -> str:
    return Path(ospath.relpath(str(target), str(from_dir))).as_posix()


def inline_code(text: str) -> str:
    parts = []
    in_tick = False
    buff = ""
    for ch in text:
        if ch == "`":
            if in_tick:
                parts.append(f"<code>{html.escape(buff)}</code>")
                buff = ""
                in_tick = False
            else:
                parts.append(html.escape(buff))
                buff = ""
                in_tick = True
        else:
            buff += ch
    if in_tick:
        parts.append("`" + html.escape(buff))
    else:
        parts.append(html.escape(buff))
    return "".join(parts)


def render_inline(text: str) -> str:
    out = []
    last = 0
    for match in LINK_RE.finditer(text):
        out.append(inline_code(text[last : match.start()]))
        link_text = inline_code(match.group(1))
        link_href = html.escape(match.group(2))
        out.append(f"<a href=\"{link_href}\">{link_text}</a>")
        last = match.end()
    out.append(inline_code(text[last:]))
    return "".join(out)


def ensure_highlighter(root: Path, tool_dir: Path) -> Path | None:
    nanoc = root / "bin" / "nanoc"
    if not nanoc.exists():
        return None
    tool_src = root / HIGHLIGHT_TOOL
    if not tool_src.exists():
        return None
    tool_dir.mkdir(parents=True, exist_ok=True)
    tool_bin = tool_dir / "nano_highlight"
    if tool_bin.exists() and tool_bin.stat().st_mtime >= tool_src.stat().st_mtime:
        return tool_bin
    cmd = ["perl", "-e", "alarm 30; exec @ARGV", str(nanoc), str(tool_src), "-o", str(tool_bin)]
    result = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True)
    if result.returncode != 0:
        return None
    return tool_bin


def highlight_nano(code: str, root: Path, tool_dir: Path) -> str:
    tool = ensure_highlighter(root, tool_dir)
    if not tool:
        return html.escape(code)
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".nano") as temp:
        temp.write(code)
        temp_path = Path(temp.name)
    try:
        result = subprocess.run([str(tool), str(temp_path)], cwd=str(root), text=True, capture_output=True)
        if result.returncode != 0:
            return html.escape(code)
        return result.stdout
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

def md_to_html(md_text: str, *, root: Path, out_dir: Path, tool_dir: Path) -> str:
    lines = md_text.splitlines()
    out: list[str] = []
    in_code = False
    code_lang = ""
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    code_lines: list[str] = []
    for line in lines:
        if not in_code and SNIPPET_MARKER_RE.match(line):
            continue
        if not in_code:
            fence = FENCE_START_RE.match(line)
            if fence:
                close_list()
                in_code = True
                code_lang = (fence.group(1) or "").strip()
                out.append(f"<pre><code class=\"language-{html.escape(code_lang)}\">")
                code_lines = []
                continue

            if line.startswith("#"):
                close_list()
                level = len(line) - len(line.lstrip("#"))
                level = max(1, min(level, 6))
                text = line[level:].strip()
                out.append(f"<h{level}>{render_inline(text)}</h{level}>")
                continue

            if line.startswith("- "):
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                item = line[2:].strip()
                out.append(f"<li>{render_inline(item)}</li>")
                continue

            if not line.strip():
                close_list()
                out.append("<div class=\"spacer\"></div>")
                continue

            close_list()
            out.append(f"<p>{render_inline(line)}</p>")
        else:
            if FENCE_END_RE.match(line):
                code_text = "\n".join(code_lines)
                if code_lang.lower() in {"nano", "nanolang"}:
                    out.append(highlight_nano(code_text, root, tool_dir))
                else:
                    out.append(html.escape(code_text))
                out.append("</code></pre>")
                in_code = False
                code_lang = ""
                continue
            code_lines.append(line)

    close_list()
    if in_code:
        code_text = "\n".join(code_lines)
        if code_lang.lower() in {"nano", "nanolang"}:
            out.append(highlight_nano(code_text, root, tool_dir))
        else:
            out.append(html.escape(code_text))
        out.append("</code></pre>")
    return "\n".join(out)


def extract_title(md_text: str, fallback: str) -> str:
    for line in md_text.splitlines():
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if level >= 1:
                return line[level:].strip()
    return fallback


def main() -> int:
    root = repo_root()
    src_dir = root / "userguide"
    out_dir = root / "build" / "userguide" / "html"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets_src = src_dir / "assets"
    assets_out = out_dir / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, assets_out, dirs_exist_ok=True)

    md_files = sorted(p for p in src_dir.rglob("*.md") if p.is_file())
    pages: list[tuple[str, str]] = []

    for md in md_files:
        rel = md.relative_to(src_dir)
        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text, rel.stem.replace("_", " "))
        body = md_to_html(md_text, root=root, out_dir=out_dir, tool_dir=out_dir.parent)
        out_path = (out_dir / rel).with_suffix(".html")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        css_href = rel_href(out_dir / "assets" / "style.css", out_path.parent)
        home_href = rel_href(out_dir / "index.html", out_path.parent)

        pages.append((title, rel.with_suffix(".html").as_posix()))

        out_path.write_text(
            "\n".join(
                [
                    "<!doctype html>",
                    "<html>",
                    "<head>",
                    "  <meta charset=\"utf-8\">",
                    f"  <title>{html.escape(title)}</title>",
                    "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
                    f"  <link rel=\"stylesheet\" href=\"{html.escape(css_href)}\">",
                    "</head>",
                    "<body>",
                    "<nav>",
                    f"  <a href=\"{html.escape(home_href)}\">NanoLang User Guide</a>",
                    "</nav>",
                    "<main>",
                    body,
                    "</main>",
                    "</body>",
                    "</html>",
                ]
            ),
            encoding="utf-8",
        )

    links = "\n".join(
        f"<li><a href=\"{html.escape(path)}\">{render_inline(title)}</a></li>" for title, path in pages
    )
    css_href = rel_href(out_dir / "assets" / "style.css", out_dir)
    (out_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "  <meta charset=\"utf-8\">",
                "  <title>NanoLang User Guide</title>",
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
                f"  <link rel=\"stylesheet\" href=\"{html.escape(css_href)}\">",
                "</head>",
                "<body>",
                "<nav>",
                "  <a href=\"index.html\">NanoLang User Guide</a>",
                "</nav>",
                "<main>",
                "<h1>NanoLang User Guide</h1>",
                "<ul>",
                links,
                "</ul>",
                "</main>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Built HTML to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
