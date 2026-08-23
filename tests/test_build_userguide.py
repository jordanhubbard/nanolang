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


if __name__ == "__main__":
    unittest.main()
