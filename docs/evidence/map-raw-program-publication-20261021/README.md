# Explicit raw ordinary map publication

I found that the prior raw qualification used only the selected-shadow API, although the contract also requires raw ordinary emission. I added a test-owned dual-path driver calling nanoisa_emit_parser_nasm and nanoisa_emit_shadows_nasm independently. It does not run semantic checking; raw declaration provenance remains explicit. The old shared shadow fixture is unchanged.

One bounded C-hosted driver build from the pinned existing C seed passed in 79.451s, with a 900s effective product-build budget and private module cache. This is a test-tool build, not a fresh compiler/bootstrap. Existing compiler source and host provider pins are retained.

The full six-method publisher corpus passed in 27.485s: 36 actual producer modules, 72 O0/O2 ASan/UBSan native executions, 27 selected forced-GC modules and nine omitted legacy modules. Raw ordinary results join the exact layout/binding comparison; raw and checked shadows remain independently compared. A seventh focused method passed in 0.090s: actual mismatched map declaration/source types preserve checked prior outputs, while both raw APIs reject with the declared-field-type diagnostic and no authority. These are six-plus-one retained terminals, not a claimed single seven-method run.

I retained the original raw mismatch probe (exit1, expected diagnostic) separately. No production policy changed and no warnings were suppressed. The default required target now builds the dual driver with C seed/Stage1/Stage2; the scoped recipe points to the retained qualified dual-driver artifact. Fresh raw stages and exact-head hosted acceptance remain open.
