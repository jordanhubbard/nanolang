#!/usr/bin/env python3
"""Build and validate the published NanoLang user guide."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "userguide"
OUTPUT = ROOT / "build/userguide/html"
GENERATED = ROOT / "build/userguide/generated"
I18N = SOURCE / "i18n"

LOCALES = (
    ("en", "English", "ltr", "en"),
    ("zh", "中文", "ltr", "zh-Hans"),
    ("hi", "हिन्दी", "ltr", "hi"),
    ("es", "Español", "ltr", "es"),
    ("ar", "العربية", "rtl", "ar"),
    ("fr", "Français", "ltr", "fr"),
)

LOCALE_LABEL = {code: label for code, label, _dir, _bcp in LOCALES}
LOCALE_DIR = {code: direction for code, _label, direction, _bcp in LOCALES}
LOCALE_BCP = {code: bcp for code, _label, _dir, bcp in LOCALES}

ACTIVE_LOCALES = [code for code, *_rest in LOCALES]


@dataclass(frozen=True)
class Page:
    source: Path
    rel_source: Path
    rel_output: Path
    title: str
    section: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def memory_hashes() -> dict[str, str]:
    path = I18N / "memory.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, text[end + 5 :]


def translation_source(page: Page, locale: str, english: str) -> tuple[str, dict[str, str]]:
    if locale == "en":
        return english, {"machine_generated": "false", "stale": "false", "fallback": "false"}
    rel = page.rel_source.as_posix()
    path = I18N / locale / rel
    hashes = memory_hashes()
    current = sha256_text(english)
    remembered = hashes.get(rel, "")
    stale = bool(remembered) and remembered != current
    if page.rel_source.parts[0] == "generated" or not path.exists():
        banner = (
            "> This page is an English fallback. I have not published a "
            f"{LOCALE_LABEL[locale]} translation of this generated reference yet.\n\n"
        )
        return banner + english, {
            "machine_generated": "true",
            "stale": "true" if stale else "false",
            "fallback": "true",
        }
    meta, body = parse_front_matter(path.read_text())
    if stale:
        body = (
            "> Stale translation: the English source changed after this draft. "
            "Treat the English edition as the authority.\n\n"
            + body
        )
        meta["stale"] = "true"
    meta.setdefault("machine_generated", "true")
    meta.setdefault("fallback", "false")
    return body, meta


def parse_nav() -> list[Page]:
    pages: list[Page] = []
    seen_sources: set[Path] = set()
    seen_outputs: set[Path] = set()
    for number, raw in enumerate((SOURCE / "nav.txt").read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 3:
            raise ValueError(f"userguide/nav.txt:{number}: expected source | title | section")
        rel_source = Path(fields[0])
        rel_output = rel_source.with_suffix(".html")
        if rel_source in seen_sources or rel_output in seen_outputs:
            raise ValueError(f"userguide/nav.txt:{number}: duplicate page {rel_source}")
        seen_sources.add(rel_source)
        seen_outputs.add(rel_output)
        pages.append(Page(SOURCE / rel_source, rel_source, rel_output, fields[1], fields[2]))
    if not pages or pages[0].rel_output != Path("index.html"):
        raise ValueError("userguide/nav.txt must start with index.md")
    return pages


def example_metadata(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines()[:20]:
        match = re.match(r"#\s*([^:]+):\s*(.*)$", line)
        if match:
            fields[match.group(1).strip()] = match.group(2).strip()
    return fields


def generate_examples() -> str:
    paths = sorted((ROOT / "examples").rglob("*.nano"))
    rows = []
    for path in paths:
        meta = example_metadata(path)
        rel = path.relative_to(ROOT).as_posix()
        name = meta.get("Example", path.stem.replace("_", " ").title())
        purpose = meta.get("Purpose", "No purpose metadata yet.")
        track = meta.get("Track", "unclassified")
        build = meta.get("Build", "unspecified")
        href = f"https://github.com/jordanhubbard/nanolang/blob/main/{rel}"
        rows.append(f"| [{name}]({href}) | `{rel}` | {track} | {build} | {purpose} |")
    return "\n".join([
        "# Examples",
        "",
        f"I have {len(paths)} NanoLang examples on disk. This table is generated from those files; I do not maintain a second count by hand.",
        "",
        "| Example | Source | Track | Build | Purpose |",
        "| --- | --- | --- | --- | --- |",
        *rows,
    ])


def module_summary(manifest: Path) -> tuple[str, str, str]:
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return (manifest.parent.name, "unknown", "Manifest could not be read.")
    return (
        str(data.get("name", manifest.parent.name)),
        str(data.get("stability", "unspecified")),
        str(data.get("summary", data.get("description", "No summary yet."))),
    )


def public_declarations(directory: Path) -> list[tuple[str, str]]:
    declarations: list[tuple[str, str]] = []
    pattern = re.compile(r"^\s*(pub\s+)?(extern\s+)?(fn|struct|enum|union|opaque\s+type|resource\s+struct)\s+([A-Za-z_][A-Za-z0-9_]*)")
    for source in sorted(directory.rglob("*.nano")):
        if source.name == "mvp.nano":
            continue
        for line in source.read_text(errors="replace").splitlines():
            match = pattern.match(line)
            if match and (match.group(1) or match.group(2)):
                declarations.append((source.relative_to(ROOT).as_posix(), line.strip()))
    return declarations


def generate_modules() -> str:
    rows: list[str] = []
    details: list[str] = []
    for directory in sorted(path for path in (ROOT / "modules").iterdir() if path.is_dir() and not path.name.startswith(".")):
        manifest = directory / "module.manifest.json"
        build_manifest = directory / "module.json"
        name, stability, summary = module_summary(manifest) if manifest.exists() else (directory.name, "unclassified", "No discovery manifest yet.")
        declarations = public_declarations(directory)
        rows.append(f"| [{name}](#{slugify(name)}) | {stability} | {'yes' if build_manifest.exists() else 'no'} | {len(declarations)} | {summary} |")
        details.extend([f"## {name}", "", summary, "", f"Source: [`modules/{directory.name}/`](https://github.com/jordanhubbard/nanolang/tree/main/modules/{directory.name})", ""])
        if declarations:
            details.extend(["| Source | Public or foreign declaration |", "| --- | --- |"])
            for source, declaration in declarations:
                escaped = declaration.replace("|", "\\|")
                details.append(f"| `{source}` | `{escaped}` |")
        else:
            details.append("No explicit `pub` or `extern` declaration was found by the catalog extractor.")
        details.append("")
    return "\n".join([
        "# Modules",
        "",
        "I generate this inventory from `modules/`, discovery manifests, build manifests, and explicit public or foreign declarations. A module can be portable in principle without being supported by every backend.",
        "",
        "| Module | Stability | Native build metadata | Declarations | Summary |",
        "| --- | --- | --- | ---: | --- |",
        *rows,
        "",
        *details,
    ])


def generate_cli() -> str:
    help_text = ""
    compiler = ROOT / "bin/nanoc_c"
    if compiler.exists():
        import subprocess
        result = subprocess.run([str(compiler), "--help"], text=True, capture_output=True, check=False)
        help_text = result.stdout + result.stderr
        help_text = help_text.replace(str(compiler), "./bin/nanoc")
    if not help_text.strip():
        help_text = "Build `bin/nanoc_c` to generate the compiler help text."
    main_source = (ROOT / "src/main.c").read_text()
    parsed_options = sorted(set(re.findall(r'"(--[a-z0-9-]+)"', main_source)))
    undocumented = [option for option in parsed_options if option not in help_text]
    audit = [
        "## Parser Audit",
        "",
        f"I found {len(parsed_options)} long option spellings in `src/main.c`.",
        "",
    ]
    if undocumented:
        audit.extend([
            "These parsed spellings are absent from the current `--help` output:",
            "",
            *[f"- `{option}`" for option in undocumented],
            "",
        ])
    else:
        audit.extend(["Every parsed long option appears in `--help`.", ""])
    return "\n".join([
        "# Compiler CLI",
        "",
        "This page embeds the help text from the compiler used for the guide build. The parser remains the authority when an old document disagrees.",
        "",
        "```text",
        help_text.rstrip(),
        "```",
        "",
        *audit,
    ])


def generate_sources() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "examples.md").write_text(generate_examples())
    (GENERATED / "builtins.md").write_text((ROOT / "docs/STDLIB.md").read_text())
    (GENERATED / "modules.md").write_text(generate_modules())
    (GENERATED / "cli.md").write_text(generate_cli())


def source_text(page: Page) -> str:
    if page.rel_source.parts[0] == "generated":
        return (GENERATED / page.rel_source.name).read_text()
    if not page.source.exists():
        raise FileNotFoundError(page.source)
    return page.source.read_text()


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_~]", "", text).lower()
    kept: list[str] = []
    for ch in text:
        kind = unicodedata.category(ch)[0]
        kept.append(ch if kind in ("L", "M", "N") else "-")
    slug = re.sub(r"-+", "-", "".join(kept)).strip("-")
    return slug or "section"


def rewrite_href(href: str, page: Page, source_to_output: dict[Path, Path]) -> str:
    split = urlsplit(href)
    if split.scheme or href.startswith(("#", "mailto:")):
        return href
    import os
    relative = Path(os.path.normpath(page.rel_source.parent / split.path))
    if relative.suffix == ".md" and relative in source_to_output:
        target_output = source_to_output[relative]
        current_dir = page.rel_output.parent
        rewritten = Path(os.path.relpath(target_output, current_dir)).as_posix()
        return rewritten + (f"#{split.fragment}" if split.fragment else "")
    return href


def render_inline(text: str, page: Page, source_to_output: dict[Path, Path]) -> str:
    text = html.escape(text, quote=False)
    link_pattern = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
    def link(match: re.Match[str]) -> str:
        label, href = match.group(1), html.unescape(match.group(2))
        rewritten = rewrite_href(href, page, source_to_output)
        if match.group(0).startswith("!"):
            return f'<img src="{html.escape(rewritten, quote=True)}" alt="{html.escape(label, quote=True)}">'
        return f'<a href="{html.escape(rewritten, quote=True)}">{label}</a>'
    text = link_pattern.sub(link, text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)
    return text


def render_markdown(markdown: str, page: Page, source_to_output: dict[Path, Path]) -> tuple[str, set[str]]:
    lines = markdown.splitlines()
    out: list[str] = []
    anchors: set[str] = set()
    slug_counts: dict[str, int] = {}
    index = 0
    in_code = False
    code_lang = ""
    code: list[str] = []

    def heading_slug(text: str) -> str:
        base = slugify(text)
        count = slug_counts.get(base, 0)
        slug_counts[base] = count + 1
        return base if count == 0 else f"{base}-{count}"

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            if not in_code:
                in_code, code_lang, code = True, line[3:].strip(), []
            else:
                out.append(f'<pre><code class="language-{html.escape(code_lang, quote=True)}">{html.escape(chr(10).join(code))}</code></pre>')
                in_code = False
            index += 1
            continue
        if in_code:
            code.append(line)
            index += 1
            continue
        if line.strip().startswith("<!--"):
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2).strip()
            anchor = heading_slug(text)
            anchors.add(anchor)
            out.append(f'<h{level} id="{anchor}">{render_inline(text, page, source_to_output)}<a class="anchor" href="#{anchor}" aria-label="Link to this section">#</a></h{level}>')
            index += 1
            continue
        if re.match(r"^\s*(---+|\*\*\*+)\s*$", line):
            out.append("<hr>")
            index += 1
            continue
        if line.startswith(">"):
            quote: list[str] = []
            while index < len(lines) and lines[index].startswith(">"):
                quote.append(re.sub(r"^>\s?", "", lines[index]))
                index += 1
            inner, _ = render_markdown("\n".join(quote), page, source_to_output)
            out.append(f"<blockquote>{inner}</blockquote>")
            continue
        list_match = re.match(r"^\s*([-+*]|\d+\.)\s+(.+)$", line)
        if list_match:
            ordered = list_match.group(1)[0].isdigit()
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while index < len(lines):
                current = re.match(r"^\s*([-+*]|\d+\.)\s+(.+)$", lines[index])
                if not current or current.group(1)[0].isdigit() != ordered:
                    break
                items.append(f"<li>{render_inline(current.group(2), page, source_to_output)}</li>")
                index += 1
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue
        if "|" in line and index + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+", lines[index + 1]):
            table_lines = [line]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            rows: list[str] = []
            for row_index, row in enumerate(table_lines):
                cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
                cell_tag = "th" if row_index == 0 else "td"
                rows.append("<tr>" + "".join(f"<{cell_tag}>{render_inline(cell, page, source_to_output)}</{cell_tag}>" for cell in cells) + "</tr>")
            out.append('<div class="table-scroll"><table>' + "".join(rows) + "</table></div>")
            continue
        if not line.strip():
            index += 1
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r"^(#{1,6})\s|^```|^>|^\s*([-+*]|\d+\.)\s+|^\s*(---+|\*\*\*+)\s*$", lines[index]):
            if "|" in lines[index] and index + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+", lines[index + 1]):
                break
            paragraph.append(lines[index].strip())
            index += 1
        out.append(f"<p>{render_inline(' '.join(paragraph), page, source_to_output)}</p>")
    if in_code:
        raise ValueError(f"{page.rel_source}: unclosed code fence")
    return "\n".join(out), anchors


def locale_output(locale: str) -> Path:
    return OUTPUT if locale == "en" else OUTPUT / locale


def locale_nav_titles(pages: list[Page], locale: str) -> dict[Path, str]:
    titles: dict[Path, str] = {}
    if locale == "en":
        return titles
    for page in pages:
        english = source_text(page)
        _body, meta = translation_source(page, locale, english)
        title = meta.get("title")
        if title:
            titles[page.rel_source] = title
    return titles


def page_href(from_locale: str, to_locale: str, from_rel: Path, to_rel: Path) -> str:
    src = locale_output(from_locale) / from_rel
    dst = locale_output(to_locale) / to_rel
    return Path(os.path.relpath(dst, src.parent)).as_posix()


SECTION_LABELS = {
    "en": {"Start": "Start", "Learn": "Learn", "Use": "Use", "Reference": "Reference"},
    "zh": {"Start": "开始", "Learn": "学习", "Use": "使用", "Reference": "参考"},
    "hi": {"Start": "शुरुआत", "Learn": "सीखें", "Use": "उपयोग", "Reference": "संदर्भ"},
    "es": {"Start": "Inicio", "Learn": "Aprender", "Use": "Usar", "Reference": "Referencia"},
    "ar": {"Start": "ابدأ", "Learn": "تعلّم", "Use": "استخدم", "Reference": "مرجع"},
    "fr": {"Start": "Début", "Learn": "Apprendre", "Use": "Utiliser", "Reference": "Référence"},
}


def navigation(pages: list[Page], current: Page, locale: str = "en", titles: dict[Path, str] | None = None) -> str:
    groups: list[str] = []
    section_names = SECTION_LABELS.get(locale, SECTION_LABELS["en"])
    for section in dict.fromkeys(page.section for page in pages):
        links = []
        for page in pages:
            if page.section != section:
                continue
            href = Path(os.path.relpath(page.rel_output, current.rel_output.parent)).as_posix()
            active = ' aria-current="page" class="active"' if page == current else ""
            label = (titles or {}).get(page.rel_source, page.title)
            links.append(f'<li><a href="{href}"{active}>{html.escape(label)}</a></li>')
        groups.append(f'<section><h2>{html.escape(section_names.get(section, section))}</h2><ul>{"".join(links)}</ul></section>')
    return "".join(groups)


def language_switcher(page: Page, locale: str) -> str:
    links = []
    for code, label, _direction, _bcp in LOCALES:
        if code not in ACTIVE_LOCALES:
            continue
        href = page_href(locale, code, page.rel_output, page.rel_output)
        current = ' aria-current="page"' if code == locale else ""
        links.append(f'<a href="{html.escape(href, quote=True)}" lang="{LOCALE_BCP[code]}"{current}>{html.escape(label)}</a>')
    return f'<nav class="langs" aria-label="Language">{"".join(links)}</nav>'


def hreflang_tags(page: Page, locale: str) -> str:
    tags = []
    for code, _label, _direction, bcp in LOCALES:
        if code not in ACTIVE_LOCALES:
            continue
        href = page_href(locale, code, page.rel_output, page.rel_output)
        tags.append(f'<link rel="alternate" hreflang="{bcp}" href="{html.escape(href, quote=True)}">')
    canonical = page_href(locale, locale, page.rel_output, page.rel_output)
    tags.append(f'<link rel="canonical" href="{html.escape(canonical, quote=True)}">')
    return "\n".join(tags)


def page_html(page: Page, body: str, pages: list[Page], locale: str = "en", titles: dict[Path, str] | None = None) -> str:
    root = Path(os.path.relpath(Path("."), page.rel_output.parent)).as_posix()
    home = Path(os.path.relpath(Path("index.html"), page.rel_output.parent)).as_posix()
    css = f"{root}/assets/style.css" if root != "." else "assets/style.css"
    lang = LOCALE_BCP[locale]
    direction = LOCALE_DIR[locale]
    return f'''<!doctype html>
<html lang="{html.escape(lang, quote=True)}" dir="{direction}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="NanoLang user guide">
<title>{html.escape(page.title)} | NanoLang</title>
<link rel="stylesheet" href="{css}">
{hreflang_tags(page, locale)}
</head>
<body>
<a class="skip-link" href="#content">Skip to content</a>
<header class="site-header"><a href="{home}">NanoLang</a><span>User Guide</span>{language_switcher(page, locale)}</header>
<div class="layout">
<nav class="sidebar" aria-label="Guide navigation">{navigation(pages, page, locale, titles)}</nav>
<main id="content">{body}</main>
</div>
</body>
</html>'''


def edition_html_files(output_root: Path, locale: str) -> list[Path]:
    other = {code for code, *_ in LOCALES if code != "en"}
    files = []
    for path in output_root.rglob("*.html"):
        rel = path.relative_to(output_root)
        if locale == "en" and rel.parts and rel.parts[0] in other:
            continue
        files.append(path)
    return files


def validate_site(pages: list[Page], anchors: dict[Path, set[str]], output_root: Path, locale: str) -> None:
    errors: list[str] = []
    expected = {page.rel_output for page in pages}
    html_files = edition_html_files(output_root, locale)
    actual = {path.relative_to(output_root) for path in html_files}
    if expected != actual:
        errors.append(f"{output_root.relative_to(ROOT)}: HTML inventory mismatch: expected {len(expected)}, found {len(actual)}")
    href_pattern = re.compile(r'href="([^"]+)"')
    id_pattern = re.compile(r'id="([^"]+)"')
    site_root = OUTPUT.resolve()
    for output in sorted(html_files):
        text = output.read_text()
        ids = id_pattern.findall(text)
        rel = output.relative_to(output_root)
        page_anchor_set = anchors.setdefault(rel, set())
        page_anchor_set.update(ids)
        if len(ids) != len(set(ids)):
            errors.append(f"{output.relative_to(ROOT)}: duplicate HTML id")
        if 'hreflang="' not in text:
            errors.append(f"{output.relative_to(ROOT)}: missing hreflang")
        if 'lang="' not in text:
            errors.append(f"{output.relative_to(ROOT)}: missing lang")
        if locale == "ar" and 'dir="rtl"' not in text:
            errors.append(f"{output.relative_to(ROOT)}: Arabic edition missing dir=rtl")
        if 'class="langs"' not in text:
            errors.append(f"{output.relative_to(ROOT)}: missing language switcher")
        for href in href_pattern.findall(text):
            split = urlsplit(html.unescape(href))
            if split.scheme or href.startswith("mailto:"):
                continue
            target = (output.parent / (split.path or output.name)).resolve()
            try:
                target.relative_to(site_root)
            except ValueError:
                errors.append(f"{output.relative_to(ROOT)}: link escapes site: {href}")
                continue
            if split.path and not target.exists():
                errors.append(f"{output.relative_to(ROOT)}: missing target: {href}")
            if split.fragment and target.suffix == ".html":
                try:
                    target_rel = target.relative_to(output_root.resolve())
                except ValueError:
                    continue
                if split.fragment not in anchors.get(target_rel, set()):
                    errors.append(f"{output.relative_to(ROOT)}: missing fragment: {href}")
    if errors:
        raise ValueError("\n".join(errors))


def build_locale(pages: list[Page], locale: str) -> dict[Path, set[str]]:
    out = locale_output(locale)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE / "assets/style.css", out / "assets/style.css")
    source_to_output = {page.rel_source: page.rel_output for page in pages}
    anchors: dict[Path, set[str]] = {}
    titles = locale_nav_titles(pages, locale)
    for page in pages:
        english = source_text(page)
        markdown, meta = translation_source(page, locale, english)
        body, page_anchors = render_markdown(markdown, page, source_to_output)
        destination = out / page.rel_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(page_html(page, body, pages, locale, titles))
        anchors[page.rel_output] = page_anchors
    print(f"Built {len(pages)} {locale} pages in {out.relative_to(ROOT)}")
    return anchors


def build(locales: list[str] | None = None) -> None:
    global ACTIVE_LOCALES
    wanted = locales or [code for code, *_ in LOCALES]
    unknown = [code for code in wanted if code not in LOCALE_DIR]
    if unknown:
        raise ValueError(f"unknown locale: {', '.join(unknown)}")
    ACTIVE_LOCALES = wanted
    pages = parse_nav()
    generate_sources()
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True)
    anchors_by_locale: dict[str, dict[Path, set[str]]] = {}
    for locale in wanted:
        anchors_by_locale[locale] = build_locale(pages, locale)
    for locale in wanted:
        validate_site(pages, anchors_by_locale[locale], locale_output(locale), locale)
        print(f"Validated {locale} edition")
    print(f"Built {len(wanted)} guide editions")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="build and validate the guide")
    parser.add_argument(
        "--locales",
        default="en,zh,hi,es,ar,fr",
        help="comma-separated locale codes",
    )
    args = parser.parse_args()
    try:
        build([part.strip() for part in args.locales.split(",") if part.strip()])
    except (OSError, ValueError) as error:
        print(f"userguide: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
