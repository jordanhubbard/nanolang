#!/usr/bin/env python3
"""Build the NanoLang developer deck from repository evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("NANOLANG_DECK_REPO", str(HERE.parents[1])))
OUT = Path(os.environ.get("NANOLANG_DECK_OUTPUT", str(HERE / "nanolang-developer-overview.pptx")))
ASSETS = HERE / "assets"
W, H = 13.333, 7.5
INK = RGBColor(0x10, 0x13, 0x17)
PANEL = RGBColor(0x23, 0x28, 0x2F)
FOG = RGBColor(0xEE, 0xF1, 0xF3)
ORANGE = RGBColor(0xFF, 0x6B, 0x35)
GREEN = RGBColor(0x76, 0xB9, 0x00)
BLUE = RGBColor(0x72, 0xB7, 0xD6)
STEEL = RGBColor(0x65, 0x70, 0x7C)


def box(slide, x, y, w, h, fill, radius=False):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def text(slide, value, x, y, w, h, size=20, color=INK, bold=False, mono=False, align=None):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.05)
    frame.margin_top = frame.margin_bottom = Inches(0.03)
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.text = value
    for paragraph in frame.paragraphs:
        if align is not None:
            paragraph.alignment = align
        for run in paragraph.runs:
            run.font.name = "Courier New" if mono else "Arial"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
    return shape


def notes(slide, lines):
    slide.notes_slide.notes_text_frame.text = "\n".join(lines)


def footer(slide, number):
    text(slide, "NANOLANG", 0.45, 7.26, 1.2, 0.15, 8, STEEL, True)
    text(slide, str(number), 12.65, 7.26, 0.25, 0.15, 8, STEEL, True, align=PP_ALIGN.RIGHT)


def title(slide, headline, subtitle, number):
    text(slide, headline, 0.65, 0.5, 11.8, 0.7, 28, INK, True)
    text(slide, subtitle, 0.68, 1.28, 11.6, 0.45, 13, STEEL)
    footer(slide, number)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    blank = prs.slide_layouts[6]

    def slide(fill=FOG):
        s = prs.slides.add_slide(blank)
        box(s, 0, 0, W, H, fill)
        return s

    mascot = ASSETS / "nanolang-mascot.png"

    s = slide(INK)
    s.shapes.add_picture(str(mascot), Inches(8.0), Inches(0.0), width=Inches(5.33), height=Inches(7.5))
    box(s, 0, 0, 9.1, H, INK)
    text(s, "NANOLANG", 0.7, 0.6, 2.0, 0.3, 13, GREEN, True)
    text(s, "I say what I mean.\nI compile myself.\nI show my evidence.", 0.7, 1.55, 7.0, 2.5, 34, FOG, True)
    text(s, "A developer's view of my syntax, compiler, NanoISA, NanoVM, and the 4.0 boundary.", 0.75, 4.45, 6.4, 0.9, 17, BLUE)
    text(s, "3.5 shipped · 4.0 begins", 0.75, 6.55, 4.0, 0.3, 13, ORANGE, True)
    notes(s, ["Authority: docs/PERSONA.md, README.md, docs/RELEASE_3.5.md.", "I describe tested behavior and label 4.0 work as future work."])

    s = slide(); title(s, "My design refuses ambiguity.", "Prefix calls, explicit boundaries, and one canonical form keep generated code readable.", 2)
    for i, (head, body) in enumerate([("PREFIX", "(f x y)"), ("TYPES", "int · float · bool · string"), ("PROOF", "shadow fn { ... }")]):
        x = 0.8 + i * 4.15
        box(s, x, 2.4, 3.5, 2.0, PANEL, True); text(s, head, x + .2, 2.7, 3.1, .3, 14, ORANGE, True); text(s, body, x + .2, 3.25, 3.1, .7, 20, FOG, True, mono=True)
    notes(s, ["Authority: docs/PERSONA.md and docs/CANONICAL_STYLE.md.", "I do not claim that every legacy surface is equally regular; the active NanoISA work is separately scoped."])

    s = slide(INK); title(s, "One source language, two execution paths.", "The C path is my native baseline. NanoISA and NanoVM make the intermediate explicit.", 3)
    for i, (head, body, color) in enumerate([(".nano", "source + shadow tests", ORANGE), ("nanoc", "generated C", BLUE), ("nano_virt", "NVM bytecode", GREEN), ("nano_vm", "decoded execution", FOG)]):
        x = .75 + i * 3.05; box(s, x, 2.6, 2.45, 1.35, PANEL, True); text(s, head, x+.15, 2.82, 2.15, .25, 15, color, True, mono=True); text(s, body, x+.15, 3.2, 2.15, .45, 12, FOG)
    text(s, "compile → verify → execute", 4.15, 5.2, 5, .35, 21, ORANGE, True, mono=True, align=PP_ALIGN.CENTER)
    notes(s, ["Authority: docs/NANOISA.md and src/nanovirt/main.c."])

    s = slide(); title(s, "NanoISA is readable bytecode, not a hidden intermediate.", "I keep the serialized module inspectable and use decoded forms only inside execution.", 4)
    box(s, .75, 2.1, 5.7, 3.7, PANEL, True); text(s, ".function main 0 0 0 int 1\n  PUSH_I64 40\n  PUSH_I64 2\n  I64_MUL\n  RET", 1.05, 2.55, 5.1, 2.2, 17, FOG, mono=True)
    box(s, 6.8, 2.1, 5.75, 3.7, INK, True); text(s, "schema → metadata → assembler\n\nThe source of truth is\nspec/nanoisa.yaml", 7.15, 2.6, 5.0, 1.8, 20, GREEN, True)
    notes(s, ["Authority: spec/nanoisa.yaml, scripts/gen_nanoisa_schema.py, src/nanoisa/assembler.c, and src/nanoisa/disassembler.c."])

    s = slide(INK); title(s, "NanoVM executes decoded instructions.", "I preserve byte offsets for diagnostics while the private cursor advances through instruction records.", 5)
    for i, label in enumerate(["decode once", "boundary map", "indexed cursor", "switch dispatch"]):
        x = .9 + i * 3.0; box(s, x, 2.65, 2.35, 1.2, PANEL, True); text(s, str(i+1), x+.15, 2.85, .35, .3, 16, ORANGE, True); text(s, label, x+.55, 2.85, 1.55, .5, 14, FOG, True)
    text(s, "NANO_VM_TRACE=1", 4.35, 5.25, 4.7, .35, 21, ORANGE, True, mono=True, align=PP_ALIGN.CENTER)
    notes(s, ["Authority: src/nanovm/vm.c, src/nanovm/vm_decode.c, and skills/nanovm-opcode-debugging/SKILL.md."])

    s = slide(); title(s, "Every function carries a shadow test.", "The test is an executable statement about behavior, not a coverage ornament.", 6)
    box(s, .8, 2.0, 5.8, 3.8, INK, True); text(s, "fn gcd(a: int, b: int) -> int {\n    ...\n}\n\nshadow gcd {\n    assert (== (gcd 48 18) 6)\n}", 1.1, 2.35, 5.2, 2.9, 16, FOG, mono=True)
    text(s, "322 NanoVM tests\n63 NanoVirt tests\n1,090 NanoISA tests", 7.15, 2.45, 4.5, 1.7, 23, GREEN, True)
    notes(s, ["Authority: CONTRIBUTING.md, userguide/part1_fundamentals/04_functions.md, and current test suites."])

    s = slide(INK); title(s, "FFI is an explicit unsafe boundary.", "Imports carry signatures. The co-process can keep foreign code outside my VM process.", 7)
    box(s, .8, 2.25, 3.0, 2.6, PANEL, True); text(s, "NanoLang", 1.05, 2.65, 2.5, .3, 18, FOG, True); text(s, "typed call", 1.05, 3.35, 2.5, .3, 15, BLUE, True)
    box(s, 5.15, 2.25, 3.0, 2.6, PANEL, True); text(s, "NanoVM", 5.4, 2.65, 2.5, .3, 18, FOG, True); text(s, "typed trap", 5.4, 3.35, 2.5, .3, 15, GREEN, True)
    box(s, 9.5, 2.25, 3.0, 2.6, PANEL, True); text(s, "nano_cop", 9.75, 2.65, 2.5, .3, 18, FOG, True); text(s, "foreign process", 9.75, 3.35, 2.5, .6, 15, ORANGE, True)
    notes(s, ["Authority: docs/EXTERN_FFI.md, src/nanovm/vm_ffi.c, and src/nanovm/cop_protocol.c.", "The FFI boundary is tested; it is not part of the formally verified NanoCore subset."])

    s = slide(); title(s, "Diagnostics are optional and cheap when disabled.", "I read runtime switches once, then keep the hot path to a cached boolean guard.", 8)
    for i, (var, body, color) in enumerate([("NANO_VM_TRACE", "opcode + stack + FFI records", ORANGE), ("NANO_PROFILE", "generated-C timing hooks", GREEN), ("--profile-isa", "structured VM counters", BLUE)]):
        y = 2.1 + i * 1.15; box(s, 1.0, y, 11.2, .8, PANEL, True); text(s, var, 1.3, y+.22, 3.0, .3, 15, color, True, mono=True); text(s, body, 4.7, y+.22, 6.8, .3, 15, FOG)
    notes(s, ["Authority: src/nanovm/vm.c, src/stdlib_runtime.c, and skills/nanovm-opcode-debugging/SKILL.md."])

    s = slide(INK); title(s, "What I measured for 3.5.", "Seven maintained workloads, 20 samples each, with the result kept as evidence.", 9)
    metrics = [("7", "workloads"), ("20", "samples each"), ("0", "failed runs"), ("all", "profiles recorded")]
    for i, (n, label) in enumerate(metrics):
        x = .8 + i * 3.05; text(s, n, x, 2.45, 2.4, .75, 38, GREEN if i != 2 else ORANGE, True, align=PP_ALIGN.CENTER); text(s, label, x, 3.35, 2.4, .35, 14, FOG, True, align=PP_ALIGN.CENTER)
    text(s, "retired · branches · calls · heap · traps · FFI", 2.0, 5.3, 9.3, .35, 18, BLUE, True, mono=True, align=PP_ALIGN.CENTER)
    notes(s, ["Authority: scripts/benchmark_nanoisa.sh, scripts/summarize_nanoisa_bench.py, and docs/RELEASE_3.5.md."])

    s = slide(); title(s, "3.5 shipped. 4.0 is a different promise.", "I keep the boundary visible so a passing benchmark is not mistaken for a complete v2 verifier.", 10)
    box(s, .8, 2.0, 5.65, 3.9, PANEL, True); text(s, "3.5 SHIPPED", 1.1, 2.35, 4.9, .3, 15, GREEN, True); text(s, "measurement\ngenerated metadata\npredecoded execution\ntail calls\nruntime diagnostics", 1.1, 2.9, 4.7, 2.2, 19, FOG, True)
    box(s, 6.9, 2.0, 5.65, 3.9, INK, True); text(s, "4.0 WORK", 7.2, 2.35, 4.9, .3, 15, ORANGE, True); text(s, "stack/type-flow verifier\nlinked callable handles\nv2 module contract\ntyped ABI and ownership\ncomputed dispatch + fusions", 7.2, 2.9, 4.7, 2.2, 19, FOG, True)
    notes(s, ["Authority: docs/RELEASE_3.5.md and docs/ROADMAP.md.", "The right column is roadmap work, not shipped behavior."])

    s = slide(INK); s.shapes.add_picture(str(mascot), Inches(8.7), Inches(.8), width=Inches(3.8), height=Inches(5.7)); text(s, "Start with the code.\nThen run the gates.", .75, 1.7, 7.4, 1.4, 34, FOG, True); text(s, "read · change · shadow-test · verify · measure", .8, 4.0, 7.5, .4, 18, ORANGE, True, mono=True); text(s, "docs/PERSONA.md · docs/ROADMAP.md · CONTRIBUTING.md", .8, 5.25, 7.4, .4, 13, BLUE, mono=True); notes(s, ["Authority: CONTRIBUTING.md, docs/PERSONA.md, and docs/ROADMAP.md."])
    footer(s, 12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    return OUT


def main() -> None:
    output = build()
    manifest = Path(os.environ.get("OBJ_DIR", str(REPO / "_build"))) / "nanolang-developer-overview" / "capability-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema": "nanolang/developer-overview@1", "slides": 12, "local_artifact": str(output)}, indent=2) + "\n")
    print(f"built 12 slides -> {output}")


if __name__ == "__main__":
    main()
