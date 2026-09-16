#!/usr/bin/env python3
"""Build the NanoLang developer narrative from repository evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("NANOLANG_DECK_REPO", str(HERE.parents[1])))
OUT = Path(os.environ.get("NANOLANG_NARRATIVE_OUTPUT", str(HERE / "nanolang-developer-overview.docx")))
PPTX = Path(os.environ.get("NANOLANG_DECK_OUTPUT", str(HERE / "nanolang-developer-overview.pptx")))


def paragraph(document: Document, value: str, *, style: str | None = None, code: bool = False) -> None:
    item = document.add_paragraph(style=style)
    item.paragraph_format.space_after = Pt(8)
    item.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    item.paragraph_format.keep_together = code
    run = item.add_run(value)
    run.font.name = "Courier New" if code else "Calibri"
    run.font.size = Pt(9 if code else 11)
    run.font.color.rgb = RGBColor(0x65, 0x70, 0x7C) if code else RGBColor(0x10, 0x13, 0x17)


def heading(document: Document, level: int, value: str) -> None:
    document.add_heading(value, level=level)


def build() -> Path:
    document = Document()
    section = document.sections[0]
    section.left_margin = section.right_margin = Inches(0.9)
    document.core_properties.title = "NanoLang: the language, compiler, and VM"
    document.core_properties.author = "NanoLang"
    document.core_properties.subject = "NanoLang 5.0 development draft — unpublished"

    heading(document, 1, "NanoLang: the language, compiler, and VM")
    heading(document, 2, "What I am")
    paragraph(document, "I am NanoLang. This is my unpublished 5.0 development draft, not a release-readiness claim. It explains my language contract, compiler paths, NanoISA bytecode, NanoVM execution, foreign-function boundary, Nano Service Interface, POSIX capability fabric, trap journal, tests, diagnostics, and unfinished work. My runtime foundations are under development. Laboratory service tests do not establish production isolation. I do not claim a kernel.")
    paragraph(document, "Authority: docs/PERSONA.md, README.md, docs/NANOISA.md, docs/NSI.md, docs/NSI_FABRIC.md, docs/NSI_EFFECTS.md, docs/NANO_EMACS.md, docs/ROADMAP.md, spec/nanoisa.yaml, docs/RELEASE_5.0.md, docs/CALLBACK_ABI.md, and the current test suites.")
    heading(document, 2, "Who this is for")
    paragraph(document, "Software developers and compiler engineers who need the technical account behind the companion deck. I describe tested behavior. Roadmap work is labelled as such.")

    heading(document, 1, "The language contract")
    heading(document, 2, "Explicit syntax and types")
    paragraph(document, "My function calls use prefix form. My function boundaries use explicit parameter and result types. My operators do not rely on a hidden precedence table. These choices make source easier to parse and review.")
    paragraph(document, (HERE / "examples/gcd.nano").read_text().strip(), code=True)
    paragraph(document, "This example checks selected nonnegative and zero inputs. It is not a theorem for every integer.")
    heading(document, 2, "Shadow tests")
    paragraph(document, "My project policy requires useful shadows. My compiler currently warns about missing tests, with exemptions documented in docs/CANONICAL_STYLE.md. A shadow checks its assertions; it does not prove every input or replace integration tests.")
    paragraph(document, "My C seed, self-hosted native driver, and bytecode CLI select root and transitive dependency shadows by default, with an explicit root-only opt-out. They run tests in separate processes under parent deadlines before publishing output. This is not a security sandbox. Source-only C emission does not execute tests.")
    heading(document, 2, "The formal core and its boundary")
    paragraph(document, "My NanoCore model states mechanized proofs for preservation, progress, determinism, semantic equivalence, and evaluator soundness under the hypotheses in formal/README.md and the theorem statements. These do not establish correspondence with my production parser, typechecker, compilers, NanoVM, FFI, NSI, or host. That correspondence requires separate evidence.")

    heading(document, 1, "The compilation paths")
    heading(document, 2, "C transpilation")
    paragraph(document, "nanoc lowers NanoLang to generated C and links the runtime libraries. This is my native execution path. Passing compilation and tests supplies evidence for the cases checked, not semantic equivalence. Full backend parity remains roadmap work.")
    heading(document, 2, "NanoISA generation")
    paragraph(document, "nano_virt lowers NanoLang to a serialized .nvm module. The active opcode metadata comes from spec/nanoisa.yaml and is generated into src/nanoisa/generated_schema.h.")
    paragraph(document, ".nano source  ->  nano_virt  ->  .nvm module  ->  nano_vm", code=True)
    heading(document, 2, "NanoVM execution")
    paragraph(document, "NanoVM decodes each function once, records instruction boundaries, and executes the decoded representation. A private indexed cursor reduces repeated byte-offset lookup while preserving byte offsets for diagnostics and traps. Bytecode is verified before it runs.")
    heading(document, 2, "Side-table debug data")
    paragraph(document, "NanoVirt records source locations in side tables. It does not emit executable debug-line instructions into normal output. --strip-debug removes those side tables when they are not wanted.")

    heading(document, 1, "Runtime boundaries")
    heading(document, 2, "Heap values and ownership")
    paragraph(document, "NanoVM values are tagged. Strings, arrays, structs, tuples, unions, hash maps, and closures are heap objects managed through retain and release operations. NanoVM collects reference cycles. Generated C already did.")
    paragraph(document, "My verifier tracks an abstract balance of explicit retain/release instructions, not object identity or complete source ownership. Tested cycle collection is not complete leak freedom. Unknown operand types are not proved safe.")
    heading(document, 2, "FFI and the co-process boundary")
    paragraph(document, "Extern declarations become typed import records. NanoVM routes calls through traps and can isolate foreign calls in nano_cop. The co-process wire fields are explicitly little-endian in the current implementation.")
    heading(document, 2, "Modules and serialized bytecode")
    paragraph(document, "The v2 NVM loader rejects duplicate or overlapping sections, directory intrusion, partial fixed-width records, trailing data, and arithmetic-overflow ranges. Feature bits fail closed.")

    heading(document, 1, "Evidence and diagnostics")
    heading(document, 2, "Unit, integration, and shadow tests")
    paragraph(document, "4.0 counted 2,632 NanoISA tests, 621 NanoVM tests, 63 NanoVirt tests, and 93 verifier tests at the v4.0.0 tag. 4.1–4.5 added NSI, fabric, catalog, Forth, nano_emacs_worker, policy, journal, and observability suites. These are historical counts, not fresh results for this draft. Configured CI workflows and local passing gates are distinct evidence; neither implies that every hosted check ran for this revision.")
    heading(document, 2, "NanoISA profiles and opcode traces")
    paragraph(document, "--profile-isa writes structured counters for retired instructions, opcode sequences, branches, calls, stack and frame depth, traps, heap traffic, and FFI traffic. NANO_VM_TRACE is read once during VM initialization and enables per-instruction records with opcode, function, offset, stack values, and FFI results.")
    heading(document, 2, "Generated-C profiling")
    paragraph(document, "Generated timing hooks read NANO_PROFILE once when the executable starts. NANO_PROFILE=0 disables a --profile build without rebuilding it. Disabled hooks do not perform environment reads or timing work at each event.")

    heading(document, 1, "Release 4.0")
    heading(document, 2, "What I shipped")
    paragraph(document, "My historical 4.0 release shipped NanoISA v2 and NanoVM v2: a regular portable instruction set, a v2 module format, modeled stack and type checks, return-shape and abstract retain/release checks, parser tests, cycle collection, and measured dispatch. See docs/RELEASE_4.0.md. This is not a whole-program safety proof.")
    heading(document, 2, "What the verifier used to miss")
    paragraph(document, "The previous verifier declared stack effects for 32 of 161 instructions and skipped the rest, including successors, then returned ok. An unknown effect is now a hard failure. Absence of evidence must not read as proof.")
    heading(document, 2, "What I measured")
    paragraph(document, "The benchmark harness times each workload once and with many iterations behind one process startup so the per-iteration cost is the difference. docs/NANOISA_MEASUREMENTS.md is the authority for every performance number, including optimizations I declined.")

    heading(document, 1, "Release 4.1–4.5")
    heading(document, 2, "Forth Core evidence")
    paragraph(document, "A NanoISA Forth session compiles colon definitions to verified bytecode. Jackson Core and Core Ext suites are vendored; make test-forth-coreext and make test-forth-jackson record what they pass. INCLUDED remains a recorded gap. I do not claim a Standard System. Authority: docs/FORTH_2012.md, docs/FORTH_STANDARD_SYSTEM.md.")
    heading(document, 2, "Catalogs and guide drafts")
    paragraph(document, "UTF-8 message catalogs exist for en, zh, hi, es, ar, and fr. Human stderr can follow the process locale. JSON and TOON stay English. User-guide drafts under userguide/i18n/ are machine-generated. I do not call the system internationalized.")
    heading(document, 2, "NSI, capabilities, and POSIX fabric")
    paragraph(document, "NSI v0 specifies ids, payloads, and compatibility with fail-closed checks. Generators emit NanoLang, Forth, Python, Rust, and C++ stubs. Checked capability APIs, shared memory, and a POSIX supervisor provide runtime foundations on an ordinary kernel. Laboratory tests do not establish unforgeability against a hostile process or production isolation. I do not claim a kernel, or a CUDA or CPython wrap. Authority: docs/NSI.md, docs/NSI_FABRIC.md, docs/NSI_TCB.md.")
    heading(document, 2, "Isolated Nano Emacs walker")
    paragraph(document, "The SDL frame does not dlopen the interpreter. Eval goes to bin/nano_emacs_worker over a length-prefixed pipe. Crash-restart keeps buffers. freeze-defun runs nano_vm as a grandchild. I do not claim GNU Emacs. Authority: docs/NANO_EMACS.md.")
    heading(document, 2, "Effects, policy, journal, and provenance")
    paragraph(document, "schema/nsi/effect_map.v0.json maps source effects, NanoISA traps, NSI methods, and capabilities. I emit an inventory, generate a deployment manifest, and reject uncovered grants. A versioned journal records trap-boundary events and replays the recorded result without calling the original service. HMAC-SHA256 authenticates a journal with a deployment key; that is not PKI. Checkpoints are sequence numbers, not heap snapshots. The journal is a tested C library, not a hook on every vm.c trap. Authority: docs/NSI_EFFECTS.md.")

    heading(document, 1, "My 5.0 candidate contract")
    paragraph(document, "I make return leave the enclosing function, run dependency shadows by default, preserve module identity and harden private native artifacts. My retained callback bridge carries signatures and lifetimes, executes on the VM owner thread and handles cancellation. Dispatch and SDL_mixer adapters use that in-process bridge; callback-bearing isolated imports remain unsupported. C-seed imported callback shadows select the same bridge. See docs/RELEASE_5.0.md and docs/CALLBACK_ABI.md.")
    paragraph(document, "The 2026-09-16 Linux finalization checkpoint records 1,730 native translator checks, 1,073 shape checks, 272,379 VM checks and 242 example programs. These are dated integration results, not acceptance of every subsequent commit. Native and VM effect repairs remain in progress. Final clean-tree tests and platform CI must establish the exact revision to release.")
    heading(document, 1, "What I have not done")
    paragraph(document, "I have not completed a Forth Standard System, reviewed human translations, GNU Emacs compatibility, a kernel, CUDA or CPython as wrapped runtimes, full backend parity, complete ownership analysis, production service isolation, or NanoISA-only self-hosted compilation with matching Stage 1/Stage 2 .nvm artifacts. See docs/ROADMAP.md; historical release scope is in docs/RELEASE_4.5.md.")
    paragraph(document, "I have implemented transitive C header dependencies in Makefile.gnu, resolving the implementation gap recorded as GitHub issue #211. The focused gate is make test-make-header-dependencies. Broader native-cache snapshot and publication requirements remain separate work.")

    document.add_page_break()
    heading(document, 1, "How to work on me")
    heading(document, 2, "Read the source and roadmap")
    paragraph(document, "Start with docs/PERSONA.md, docs/ROADMAP.md, docs/RELEASE_5.0.md, userguide/guide/08_secure_runtime.md, the relevant source symbols, and the matching tests. Do not turn a roadmap sentence into a feature claim.")
    heading(document, 2, "Run the gates")
    paragraph(document, "make test\nmake test-nsi test-nsi-cap test-nsi-fabric test-nsi-policy test-nsi-journal test-nsi-obs\nmake test-nano-emacs-worker\nmake release-docs-check", code=True)
    paragraph(document, "I say what I mean, I show what I tested, and I leave the unproved boundary visible.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUT)
    return OUT


def main() -> None:
    output = build()
    if not PPTX.is_file():
        raise SystemExit(f"presentation artifact missing: {PPTX}")
    manifest = Path(os.environ.get("OBJ_DIR", str(REPO / "_build"))) / "nanolang-developer-overview" / "capability-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    from pptx import Presentation
    manifest.write_text(json.dumps({"schema": "nanolang/developer-document-pair@1",
        "slides": len(Presentation(str(PPTX)).slides), "local_artifact": str(PPTX.resolve()),
        "narrative": str(output.resolve())}, indent=2) + "\n")
    print(f"built narrative -> {output}")


if __name__ == "__main__":
    main()
