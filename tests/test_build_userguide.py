import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "build_userguide", ROOT / "scripts/build_userguide.py"
)
build_userguide = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = build_userguide
SPEC.loader.exec_module(build_userguide)


class UserGuideBuildTests(unittest.TestCase):
    def setUp(self):
        self.page = build_userguide.Page(
            ROOT / "userguide/guide/01_getting_started.md",
            Path("guide/01_getting_started.md"),
            Path("guide/01_getting_started.html"),
            "Getting Started",
            "Learn",
        )
        self.outputs = {
            Path("guide/01_getting_started.md"): Path("guide/01_getting_started.html"),
            Path("guide/02_language.md"): Path("guide/02_language.html"),
        }

    def test_inline_markup_and_internal_link(self):
        rendered = build_userguide.render_inline(
            "Read **this** and [`code`](02_language.md#calls).",
            self.page,
            self.outputs,
        )
        self.assertIn("<strong>this</strong>", rendered)
        self.assertIn('href="02_language.html#calls"', rendered)
        self.assertIn("<code>code</code>", rendered)

    def test_operator_inside_code_is_not_emphasis(self):
        rendered = build_userguide.render_inline(
            "`2 + 3 * 4`", self.page, self.outputs
        )
        self.assertEqual(rendered, "<code>2 + 3 * 4</code>")

    def test_markdown_structures(self):
        source = """# Page

1. first
2. second

> quoted

| A | B |
| --- | --- |
| one | two |

---
"""
        rendered, anchors = build_userguide.render_markdown(
            source, self.page, self.outputs
        )
        self.assertIn('<h1 id="page">', rendered)
        self.assertIn("<ol>", rendered)
        self.assertIn("<blockquote>", rendered)
        self.assertIn("<table>", rendered)
        self.assertIn("<hr>", rendered)
        self.assertIn("page", anchors)

    def test_unclosed_fence_fails(self):
        with self.assertRaisesRegex(ValueError, "unclosed code fence"):
            build_userguide.render_markdown("```nano\nfn main", self.page, self.outputs)

    def test_navigation_has_one_current_page(self):
        pages = build_userguide.parse_nav()
        rendered = build_userguide.navigation(pages, pages[0])
        self.assertEqual(rendered.count('aria-current="page"'), 1)

    def test_generated_inventory_matches_disk(self):
        generated = build_userguide.generate_examples()
        count = len(list((ROOT / "examples").rglob("*.nano")))
        self.assertIn(f"I have {count} NanoLang examples", generated)

    def test_slugify_keeps_cjk_and_arabic(self):
        self.assertIn("开始", build_userguide.slugify("开始"))
        self.assertIn("البداية", build_userguide.slugify("البداية"))

    def test_slugify_keeps_devanagari_matras(self):
        slug = build_userguide.slugify("NanoLang उपयोगकर्ता मार्गदर्शिका")
        self.assertIn("उपयोगकर्ता", slug)
        self.assertIn("मार्गदर्शिका", slug)

    def test_translation_source_zh_index(self):
        pages = build_userguide.parse_nav()
        index = pages[0]
        english = index.source.read_text()
        body, meta = build_userguide.translation_source(index, "zh", english)
        self.assertEqual(meta.get("fallback"), "false")
        self.assertIn("用户指南", body)
        self.assertIn("```nano", body)
        self.assertIn("fn factorial", body)

    def test_stale_translation_banner(self):
        pages = build_userguide.parse_nav()
        index = pages[0]
        english = index.source.read_text()
        original = build_userguide.memory_hashes
        build_userguide.memory_hashes = lambda: {index.rel_source.as_posix(): "deadbeef"}
        try:
            body, meta = build_userguide.translation_source(index, "zh", english)
        finally:
            build_userguide.memory_hashes = original
        self.assertEqual(meta.get("stale"), "true")
        self.assertIn("Stale translation", body)

    def test_translated_nav_titles_on_index(self):
        pages = build_userguide.parse_nav()
        titles = {
            pages[0].rel_source: "首页",
            Path("guide/01_getting_started.md"): "入门",
        }
        html = build_userguide.page_html(pages[0], "<p>x</p>", pages, "zh", titles)
        self.assertIn("入门", html)
        self.assertNotIn(">Getting Started<", html)

    def test_language_switcher_keeps_page(self):
        page = self.page
        html = build_userguide.language_switcher(page, "en")
        self.assertIn("zh/guide/01_getting_started.html", html)
        self.assertIn('lang="ar"', html)
        self.assertIn('aria-current="page"', html)
        arabic = build_userguide.page_html(page, "<p>x</p>", [page], "ar")
        self.assertIn('dir="rtl"', arabic)
        self.assertIn('hreflang="zh-Hans"', arabic)
        self.assertIn('class="langs"', arabic)

    def test_memory_tracks_published_english(self):
        pages = [
            "index.md",
            "guide/01_getting_started.md",
            "guide/06_tools_and_backends.md",
        ]
        hashes = build_userguide.memory_hashes()
        for rel in pages:
            current = build_userguide.sha256_text((ROOT / "userguide" / rel).read_text())
            self.assertEqual(hashes[rel], current, rel)

    def test_css_font_fallback_and_mobile(self):
        css = (ROOT / "userguide/assets/style.css").read_text()
        self.assertIn("Noto Sans SC", css)
        self.assertIn("Noto Sans Devanagari", css)
        self.assertIn("Noto Naskh Arabic", css)
        self.assertIn('html[dir="rtl"]', css)
        self.assertIn("unicode-bidi: isolate", css)
        self.assertIn("@media (max-width: 760px)", css)
        self.assertIn(".langs", css)


if __name__ == "__main__":
    unittest.main()
