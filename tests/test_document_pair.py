"""I reject broken document pairs instead of trusting a manifest's claims."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from scripts.verify_document_pair import NS, REL, verify

ROOT = Path(__file__).resolve().parents[1]


class DocumentPairAcceptance(unittest.TestCase):
    def test_retained_deck_states_shadow_and_verifier_boundaries(self):
        with ZipFile(ROOT / "docs/presentation/nanolang-developer-overview.pptx") as archive:
            def text(part):
                return "\n".join(node.text or "" for node in ET.fromstring(archive.read(part)).findall(".//a:t", NS))
            second = text("ppt/slides/slide2.xml")
            self.assertIn("TESTS", second)
            self.assertNotIn("PROOF", second)
            shadows = text("ppt/slides/slide10.xml")
            for claim in ("project policy", "warnings", "Exemptions", "tested by default", "source-only"):
                self.assertIn(claim, shadows)
            verifier = text("ppt/slides/slide6.xml")
            for boundary in ("not whole-program proof", "not object identity", "unknown stays unknown"):
                self.assertIn(boundary, verifier)
            self.assertIn("5.0 DRAFT", text("ppt/slides/slide1.xml"))
            source = (ROOT / "docs/presentation/examples/gcd.nano").read_text().strip()
            slide = ET.fromstring(archive.read("ppt/slides/slide10.xml"))
            frames = ["\n".join("".join(node.text or "" for node in p.findall(".//a:t", NS))
                                  for p in body.findall("a:p", NS))
                      for body in slide.findall(".//p:txBody", NS)]
            self.assertIn(source, frames)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nano-document-pair-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.manifest = self.directory / "manifest.json"
        self.data = {"schema": "nanolang/developer-document-pair@1", "slides": 1,
                     "local_artifact": "deck.pptx", "narrative": "story.docx"}
        self.parts = {
            "ppt/presentation.xml": f'<p:presentation xmlns:p="{NS["p"]}" xmlns:r="{NS["r"]}">'
                '<p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>'
                '<p:sldSz cx="1000" cy="1000"/></p:presentation>',
            "ppt/_rels/presentation.xml.rels": self.rels("slide", "slides/slide1.xml"),
            "ppt/slides/_rels/slide1.xml.rels": self.rels("notesSlide", "../notesSlides/notesSlide1.xml"),
            "ppt/slides/slide1.xml": f'<p:sld xmlns:p="{NS["p"]}" xmlns:a="{NS["a"]}">'
                '<p:cSld><p:spTree>' + self.shape(10, 10) + '</p:spTree></p:cSld></p:sld>',
            "ppt/notesSlides/notesSlide1.xml": f'<p:notes xmlns:p="{NS["p"]}" xmlns:a="{NS["a"]}">'
                '<p:sp><p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr>'
                '<p:txBody><a:p><a:r><a:t>I describe tested behavior.</a:t></a:r></a:p></p:txBody></p:sp></p:notes>',
        }
        self.words = {
            "word/styles.xml": f'<w:styles xmlns:w="{NS["w"]}">' + ''.join(
                f'<w:style w:styleId="H{i}"><w:name w:val="heading {i}"/></w:style>' for i in range(1, 8)) + '</w:styles>',
            "word/document.xml": f'<w:document xmlns:w="{NS["w"]}"><w:body>'
                '<w:p><w:pPr><w:pStyle w:val="H1"/></w:pPr><w:r><w:t>I explain my language.</w:t></w:r></w:p>'
                '</w:body></w:document>',
        }

    @staticmethod
    def rels(kind, target):
        return f'<Relationships xmlns="{REL}"><Relationship Id="rId1" Type="{NS["r"]}/{kind}" Target="{target}"/></Relationships>'

    @staticmethod
    def shape(x, y, text="I test my code."):
        return f'<p:sp><p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="100" cy="100"/></a:xfrm></p:spPr>' \
               f'<p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>'

    def write(self):
        self.manifest.write_text(json.dumps(self.data))
        for filename, parts in (("deck.pptx", self.parts), ("story.docx", self.words)):
            with ZipFile(self.directory / filename, "w") as package:
                for name, content in parts.items():
                    package.writestr(name, content)

    def test_complete_pair_uses_actual_counts_without_claiming_visual_qa(self):
        self.write()
        result = verify(self.manifest)
        self.assertTrue(result["accepted"], result)
        self.assertEqual((result["slides"], result["headings"]), (1, 1))
        self.assertEqual(result["visual_qa"], "not performed")

    def test_manifest_missing_artifacts_wrong_count_or_bad_archive_fails(self):
        for key, value in (("slides", 2), ("slides", True), ("local_artifact", "missing"),
                           ("narrative", ""), ("schema", "unknown")):
            with self.subTest(key=key, value=value):
                original = self.data[key]
                self.data[key] = value
                self.write()
                self.assertFalse(verify(self.manifest)["accepted"])
                self.data[key] = original
        self.write()
        (self.directory / "deck.pptx").write_bytes(b"not a package")
        self.assertFalse(verify(self.manifest)["accepted"])

    def test_geometry_overlap_and_unsupported_transforms_fail(self):
        key = "ppt/slides/slide1.xml"
        original = self.parts[key]
        for value in (original.replace('x="10"', 'x="950"'),
                      original.replace('</p:spTree>', self.shape(50, 50) + '</p:spTree>'),
                      original.replace('<a:xfrm>', '<a:xfrm rot="60000">'),
                      original.replace('cx="100"', 'cx="0"'),
                      original.replace('</p:spTree>', '<p:grpSp/></p:spTree>')):
            with self.subTest(value=value):
                self.parts[key] = value
                self.write()
                self.assertFalse(verify(self.manifest)["accepted"])
        self.parts[key] = original.replace('</p:spTree>', self.shape(110, 10) + '</p:spTree>')
        self.write()
        self.assertTrue(verify(self.manifest)["accepted"])

    def test_empty_notes_and_heading_skips_fail(self):
        key = "ppt/notesSlides/notesSlide1.xml"
        original = self.parts[key]
        self.parts[key] = original.replace('I describe tested behavior.', '')
        self.write()
        self.assertFalse(verify(self.manifest)["accepted"])
        self.parts[key] = original
        document = self.words["word/document.xml"]
        for level in (2, 7):
            self.words["word/document.xml"] = document.replace('w:val="H1"', f'w:val="H{level}"')
            self.write()
            self.assertFalse(verify(self.manifest)["accepted"])

    def test_placeholder_and_credential_patterns_are_rejected_without_echoing_values(self):
        key = "word/document.xml"
        original = self.words[key]
        for value in ("{{UNRESOLVED}}", "TODO", "Bearer " + "x" * 30, "ghp_" + "a" * 36):
            with self.subTest(value=value):
                self.words[key] = original.replace('I explain my language.', value)
                self.write()
                result = verify(self.manifest)
                self.assertFalse(result["accepted"])
                self.assertNotIn(value, json.dumps(result))

    def test_cli_overwrites_stale_acceptance_and_exits_nonzero(self):
        self.write()
        report = self.directory / "acceptance.json"
        command = [sys.executable, str(ROOT / "scripts/verify_document_pair.py"),
                   "--manifest", str(self.manifest), "--json", str(report)]
        first = subprocess.run(command, capture_output=True, timeout=10)
        self.assertEqual(first.returncode, 0, first.stderr)
        (self.directory / "story.docx").unlink()
        second = subprocess.run(command, capture_output=True, timeout=10)
        self.assertNotEqual(second.returncode, 0)
        self.assertFalse(json.loads(report.read_text())["accepted"])

    def test_retained_artifacts_are_inspected_not_the_builder_source(self):
        self.data.update(slides=16, local_artifact=str(ROOT / "docs/presentation/nanolang-developer-overview.pptx"),
                         narrative=str(ROOT / "docs/presentation/nanolang-developer-overview.docx"))
        self.manifest.write_text(json.dumps(self.data))
        result = verify(self.manifest)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result.get("slides"), 16, result)
        self.assertGreater(result.get("headings", 0), 0, result)

    def test_real_regeneration_invalidates_prior_acceptance_on_authoring_failure(self):
        toolchain = ROOT / "_build/doc-toolchain"
        if not (toolchain / "bin/python3").is_file():
            self.skipTest("I need make doc-toolchain-bootstrap for real authoring acceptance")
        build = self.directory / "nanolang-developer-overview"
        (self.directory / "doc-toolchain").symlink_to(toolchain, target_is_directory=True)
        env = {**os.environ, "OBJ_DIR": str(self.directory),
               "NANOLANG_DECK_SKIP_RENDER": "1", "NANOLANG_DECK_OUTPUT": str(self.directory / "deck.pptx"),
               "NANOLANG_NARRATIVE_OUTPUT": str(self.directory / "story.docx")}
        command = ["bash", str(ROOT / "docs/presentation/regenerate_python.sh")]
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads((build / "acceptance.json").read_text())["accepted"])
        env["NANOLANG_NARRATIVE_OUTPUT"] = str(self.directory / "missing-parent" / "story.docx")
        (self.directory / "missing-parent").write_text("I am a file, not a directory.")
        result = subprocess.run(command, env=env, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(json.loads((build / "acceptance.json").read_text())["accepted"])

    def test_placeholder_scan_preserves_paragraph_boundaries_and_joins_runs(self):
        key = "word/document.xml"
        original = self.words[key]
        self.words[key] = original.replace('</w:body>', '<w:p><w:r><w:t>TO</w:t></w:r>'
            '<w:r><w:t>DO</w:t></w:r></w:p><w:p><w:r><w:t>Next</w:t></w:r></w:p></w:body>')
        self.write()
        self.assertFalse(verify(self.manifest)["accepted"])

    def test_missing_external_or_escaping_slide_relationship_is_rejected(self):
        key = "ppt/_rels/presentation.xml.rels"
        original = self.parts[key]
        for value in (original.replace('rId1', 'rId2'),
                      original.replace('Target=', 'TargetMode="External" Target='),
                      original.replace('slides/slide1.xml', '../../outside.xml')):
            with self.subTest(value=value):
                self.parts[key] = value
                self.write()
                self.assertFalse(verify(self.manifest)["accepted"])


if __name__ == "__main__":
    unittest.main()
