#!/usr/bin/env python3

import html
import re
import shutil
from pathlib import Path
from os import path as ospath


FENCE_START_RE = re.compile(r"^```\s*(\w+)?\s*$")
FENCE_END_RE = re.compile(r"^```\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel_href(target: Path, from_dir: Path) -> str:
    return Path(ospath.relpath(str(target), str(from_dir))).as_posix()


def md_to_html(md_text: str) -> str:
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

    for line in lines:
        if not in_code:
            fence = FENCE_START_RE.match(line)
            if fence:
                close_list()
                in_code = True
                code_lang = (fence.group(1) or "").strip()
                out.append(f"<pre><code class=\"language-{html.escape(code_lang)}\">")
                continue

            if line.startswith("#"):
                close_list()
                level = len(line) - len(line.lstrip("#"))
                level = max(1, min(level, 6))
                text = line[level:].strip()
                out.append(f"<h{level}>{html.escape(text)}</h{level}>")
                continue

            if line.startswith("- "):
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                item = line[2:].strip()
                item = LINK_RE.sub(
                    lambda m: f"<a href=\"{html.escape(m.group(2))}\">{html.escape(m.group(1))}</a>",
                    item,
                )
                out.append(f"<li>{item}</li>")
                continue

            if not line.strip():
                close_list()
                out.append("<div class=\"spacer\"></div>")
                continue

            close_list()
            escaped = html.escape(line)
            escaped = LINK_RE.sub(
                lambda m: f"<a href=\"{html.escape(m.group(2))}\">{html.escape(m.group(1))}</a>",
                escaped,
            )
            out.append(f"<p>{escaped}</p>")
        else:
            if FENCE_END_RE.match(line):
                out.append("</code></pre>")
                in_code = False
                code_lang = ""
                continue
            out.append(html.escape(line))

    close_list()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


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
        title = rel.stem.replace("_", " ")
        md_text = md.read_text(encoding="utf-8")
        body = md_to_html(md_text)
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
        f"<li><a href=\"{html.escape(path)}\">{html.escape(title)}</a></li>" for title, path in pages
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
