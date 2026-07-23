# Remaining TODOs and Next Steps

**Last Updated:** July 23, 2026
**Status:** Reconciled against implemented reality. Self-hosting is complete (3-stage
bootstrap via `make bootstrap`); see `docs/ROADMAP.md` for authoritative status.

---

## Code TODOs

_No open source TODOs are tracked here. Previously listed items have been resolved:_

- ~~**Function Call Arguments Parsing** (`src/interpreter_main.c`)~~ — obsolete: the
  standalone `src/interpreter_main.c` no longer exists; the runtime is now organized
  around `src/eval.c` and related modules.
- ~~**Array Element Type Storage** (`src/parser.c`)~~ — done: array element types are
  now stored and propagated (`parse_type_with_element`, `element_type` fields in
  `src/parser.c`); the old "for now just return TYPE_ARRAY" placeholder is gone.

To find live source TODOs, search the tree directly (e.g. `grep -rn "TODO" src/`).

---

## Self-Hosting Roadmap ✅ COMPLETE

Self-hosting is **done**, not pending. The compiler is written in nanolang under
`src_nano/` (lexer, parser, typecheck, transpiler, and drivers) and compiles itself
through a verified 3-stage bootstrap.

- ✅ **Phase 1 — Essential Features** (structs, enums, dynamic lists, file I/O,
  string ops, system execution)
- ✅ **Phase 2 — Compiler Components in nanolang** — lexer, parser, type checker,
  transpiler, and main driver all live in `src_nano/`.
- ✅ **Phase 3 — Bootstrap** — the 3-stage bootstrap runs via `make bootstrap` and
  reaches a verified fixed point.

Authoritative status lives in `docs/ROADMAP.md` (Phase 8: Self-Hosting COMPLETE).
Component-level parity notes are tracked in `planning/STAGE2_STATUS.md`.

---

## Documentation TODOs

_No open documentation TODOs are tracked here._ The previously listed status docs
(`SELF_HOSTING_CHECKLIST.md`, `SELF_HOSTING_IMPLEMENTATION_PLAN.md`,
`SELF_HOSTING_REQUIREMENTS.md`, `SELF_HOSTING_FEATURE_GAP.md`) no longer exist in the
tree; self-hosting status is consolidated into `docs/ROADMAP.md`.

---

## Feature Enhancements (Future)

These are aspirational, not committed work. Cross-check against `docs/ROADMAP.md`
before starting, since several have since landed.

### Language Features
- [x] Pattern matching (`match`) — implemented
- [x] Modules/imports — implemented
- [x] Tuples — implemented
- [x] Generics — implemented
- [ ] Richer array/list combinators (map, filter, reduce, slice) — partial
- [ ] `Result`-style error handling ergonomics — partial

### Tooling
- [x] REPL — implemented
- [x] Language server (LSP) — implemented (`src/lsp_server.c`)
- [x] Debugger integration — DAP server present (`src/dap_server.c`)
- [ ] Package manager — see `planning/PACKAGE_MANAGER_DESIGN.md`
- [ ] Documentation generator improvements — `src/docgen*.c` exists

### Optimizations
- [x] Constant folding — implemented (`src/fold_constants.c`)
- [x] Dead code elimination — implemented (`src/dce_pass.c`)
- [ ] Tail call optimization
- [ ] Inlining
- [ ] LLVM backend maturity — `src/llvm_backend.c` exists; verify coverage

---

## Notes

- **Self-Hosting:** ✅ Complete — verified 3-stage bootstrap (`make bootstrap`).
- **Formal Verification:** ✅ Core proven in Coq (see `docs/ROADMAP.md`, Phase 11).
- This file intentionally tracks only genuinely-open work. When in doubt, treat
  `docs/ROADMAP.md` as the source of truth for status.
