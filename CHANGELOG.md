# Changelog

All notable changes to NanoLang are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- 4.6 Nano Scheme laboratory frontend: lexical scope, closures, pairs,
  named tail calls, and session `define` compiled to verified NanoISA.
  Continuations, macros, and `set!` fail closed. `docs/SCHEME.md`,
  `make test-scheme`.
- 4.6 Nano ML laboratory frontend: Hindley-Milner inference, ADTs,
  exhaustive `case`, curried `fun`/`fn`, and `signature` compiled to
  verified NanoISA. `ref`, `exception`, and `:=` fail closed.
  `docs/ML.md`, `make test-ml`.
- 4.6 Nano Actor laboratory frontend: isolated NanoVM contexts, typed
  mailboxes, monitors, links, one_for_one supervision, `after 0`,
  cancel, and hot replace compiled to verified NanoISA handlers.
  Remote spawn and `cap:` fail closed. Phase 18 transport is not
  wired. `docs/ACTOR.md`, `make test-actor`.
- 4.6 Nano Dataflow laboratory frontend: typed nodes, bounded streams,
  backpressure, explicit effects, interned feed journals, replay,
  cancel, retry, and fifo/reverse determinism compiled to verified
  NanoISA. `place remote` fails closed. Phase 18 transport is not
  wired. `docs/DATAFLOW.md`, `make test-dataflow`.
- 4.6 Nano Object laboratory frontend: message send, identity, mutable
  slots, `classof`/`slots`, live `replace`, `extend`, `handle`/`sendvia`,
  and a host inline cache compiled to verified NanoISA methods. No
  cache opcodes. `docs/OBJECT.md`, `make test-object`.
- 4.6 Nano Shell laboratory frontend: typed i64 pipelines, `parse` as
  the text adapter, `need` capabilities, and `cancel`. Granted caps
  still refuse host files, processes, networks, services, streams, and
  remote execution. I do not administer service graphs.
  `docs/SHELL.md`, `make test-shell`.
- 4.6 Nano Logic laboratory frontend: bounded Datalog facts, Horn
  rules, ground queries, and a deterministic least fixed-point.
  Integer unification is `lg_unify` / `I64_EQ`. Choice points and
  tabling stay out. Policy is the restricted `grant`/`allow` profile.
  `docs/LOGIC.md`, `make test-logic`.
- 5.0 Cut A pin: `src_nano/compiler/nanoisa_codegen.nano` emits NanoISA
  assembly for integer `add`/`main`/`choose`/`loop_sum`, string
  `greeting`/`glue`, array `len3`/`first`, record `getx`, bool
  `is_pos`, bool literals `yes`/`no`, `invert`, `both`, `either`,
  `pick`, `say`, `shout`, `mutter`, `prove`, `grow`, `has_hi`,
  `digits`, `names`, `head_s`, `same`, `diff`, `via_at`, `slen`,
  `slice`, `blank_l`, `grow_l`, `ch`, `blank_s`, `grow_s`, `get_s`,
  `blank_t`, `grow_t`, `get_v`, `grow_lex`, `has_pre`, `has_suf`,
  `put_l`, `put_t`, `put_s`, `upto`, `quiet`, `via_quiet`, `origin`,
  `via_o`, `make_tok`, `via_tok`, `ones`, `via_ones`, `new_l`,
  `via_new_l`, `one_t`, `via_one_t`, `one_lex`, `via_one_lex`,
  `in_az`, `via_az`, `tag`, `via_tag`, `tok_mod`, `via_tok_mod`,
  `imp_add`, `via_imp_add`, `via_raw`, and `via_cwd`.
  `make test-nanoisa-src-nano`
  compares function bytecode with the C seed, including `if`,
  `while`, `PUSH_STR`, `STR_CONCAT`, `ARR_LITERAL`, `ARR_LEN`,
  `ARR_GET`, `ARR_PUSH` of `array<int>` and `array<string>`,
  `AGG_PACK`, `AGG_GET`, `bool` results, `PUSH_BOOL`, `BOOL_NOT`,
  `BOOL_AND`, `BOOL_OR`, `cond` as `JMP_FALSE`/`JMP` with one `RET`,
  `PRINT`, `PRINTLN`, `ASSERT`, `STR_CONTAINS`, `CAST_STRING` of
  i64, `EQ`/`NE` of strings, `at` as `ARR_GET`, `str_length` as
  `STR_LEN`, `str_substring` as `STR_SUBSTR`, `list_int_new` as
  `ARR_NEW 1`, void `list_int_push` as `ARR_PUSH` then `POP`,
  `list_int_get` as `ARR_GET`, `char_at` as `STR_CHAR_AT`,
  `list_string_new` as `ARR_NEW 1`, void `list_string_push` as
  `ARR_PUSH` then `POP`, `list_string_get` as `ARR_GET`, mixed
  int/string records as `AGG_PACK`/`AGG_GET`, `List<Tok>` as
  `ARR_NEW 1` / `ARR_PUSH` then `POP` / `ARR_GET`, and `LexerToken`
  (`token_type`/`value`/`line`/`column`) plus `List<LexerToken>` as
  `get_v`/`grow_lex`, `str_starts_with` as `STR_STARTS_WITH`, and
  `str_ends_with` as `STR_ENDS_WITH`, `list_int_set` as `ARR_SET`
  then `POP`, `list_Tok_set` as `ARR_SET` then `POP`,
  `list_string_set` as `ARR_SET` then `POP`, `for` over
  `array<int>` as `LT`, and `void` as `RET` with no value
  (`CALL quiet` does not `POP`), and pin-record results as ISA tag
  `struct` (`origin` / `via_o`), and mixed int/string record results
  (`make_tok` / `via_tok`), and array results as ISA tag `array`
  (`ones` / `via_ones`), and `List<int>` results (`new_l` /
  `via_new_l`), and `List<Tok>` results (`one_t` / `via_one_t`),
  and `List<LexerToken>` results (`one_lex` / `via_one_lex`), and
  i64 `>=`/`<=` (`in_az` / `via_az`), and `Enum.Variant` as
  `ENUM_VAL` (`tag` / `via_tag`), and imported `Enum.Variant`
  (`tok_mod` / `via_tok_mod`; local enums stay first in `def_idx`).
  One-level nested pin-record field types so `List<CompilerDiagnostic>`
  parameters type-check. Imported functions (`imp_add` / `via_imp_add`)
  `CALL`/`TAIL_CALL` by name. `unsafe` blocks emit inner statements
  (`via_raw`).   `getcwd` is `CALL_EXTERN` `vm_getcwd` (`via_cwd`).
  Nested pin-record literals (`nest_d` / `via_nest`) match the C seed.
  `nvm2c` still refuses nested records. Bool and `List<int>` record
  fields (`flag_yes` / `via_flag`, `empty_bag` / `via_bag`) match the
  C seed. String `+` of a field or `int_to_string` (`glue_field` /
  `glue_digits`) and `str_concat` as `STR_CONCAT` (`glue_sc`) match
  the C seed. `string_to_int` is `CAST_INT` (`parse_n` /
  `via_parse_n`). Nested field access is chained `AGG_GET`
  (`nest_line` / `via_nest_line`). Module-qualified call
  `(ImpMod.imp_add 40 2)` is `TAIL_CALL imp_add` (`via_q_add`).
  `parser.nano` emits 297 functions and that nasm assembles.
  `array<Loc>` (`empty_locs` / `via_empty_locs` / `one_loc` /
  `via_one_loc`) matches the C seed. HashMap results (`blank_hm` /
  `via_blank_hm` / `put_hm` / `via_put_hm`) match the C seed:
  `map_new` is `HM_NEW 5 1`, statement `map_put` is `HM_SET` then
  `POP`. Pin-record params of `array<Sym>` (`n_syms` / `via_n_syms`)
  match the C seed; a pin record may have `array<pin-record>` fields.
  Interned quotes (`quoted` / `via_quoted`) match the C seed; `.string`
  payloads escape `"`, `\`, newline, tab, and CR. `typecheck.nano`
  emits 401 functions. `nanoisa asm` names `Duplicate label: L1024`.
  `std/fs.nano` resolves through `modules/` like the C seed.
  Transitive imports register `LexerTokenType` from
  `compiler_schema.nano`. Imported structs register `LexerToken`.
  `STR_SPLIT`
  and `STR_REPLACE`
  stay refused. String operands
  are compared by content. I still pretty-print C to build the
  compiler. When I refuse a program, `nanoisa_emit` prints `nisa_err`.
  A library file gets a synthetic `main` (`PUSH_I64 0` then `RET`)
  like the C seed. `;` and `#` inside a quoted `.string` stay payload;
  `lexer.nano` nasm assembles.
- `bin/nvm2c` translates a verified `.nvm` to structured C11. I am a
  host tool, not a compiler phase. `make nvm2c`, `make test-nvm2c`.
  Generated C does not name `nano_vm`. The closed subset includes i64
  comparisons, `JMP`/`JMP_FALSE`, and `TAIL_CALL`. `choose` and
  `loop_sum` run as native C. Goto is the translator fallback.
  `PUSH_STR`, `STR_CONCAT`, and `STR_LEN` run as native C: `greeting`
  and `glue` exit with string length and do not name `nano_vm`.
  Embedded NULs and `STR_TRIM` stay refused. `ARR_LITERAL`,
  `ARR_GET`, and `ARR_LEN` run as native C: `len3` and `first` exit
  with length and the first element. Nested arrays and `ARR_SET`
  stay refused. `AGG_PACK` and `AGG_GET` run as native C: `getx`
  exits with field 0 of an `int` record. Nested records, variants,
  tuples, and `AGG_SET` stay refused. `bool` results are i64 0/1:
  `is_pos` runs as native C. `PUSH_BOOL`, `BOOL_NOT`, `BOOL_AND`, and
  `BOOL_OR` run as native C: `yes`, `invert`, `both`, and `either`
  exit without a VM process. `pick` (`cond` join) runs as native C:
  both arms share one `RET`. `PRINT` and `PRINTLN` run as native C:
  `say` writes `7`, `shout` writes `7` with a newline, `mutter`
  writes an empty string. Printing arrays stays refused. `ASSERT`
  runs as native C: `prove(true)` exits 0 and `prove(false)` aborts
  without a VM process. `ARR_PUSH` of `array<int>` runs as native C:
  `grow` exits 2. `STR_CONTAINS` runs as
  native C: `has_hi("hi")` exits 1 and `has_hi("no")` exits 0.
  `CAST_STRING` of i64 runs as native C: `digits(7)` has length 1.
  `ARR_LITERAL` tag 5, `ARR_PUSH`, `ARR_GET`, and `ARR_LEN` of
  `array<string>` run as native C: `names` exits 2 and `head_s`
  returns `"hi"`. Nested arrays stay refused. `EQ`/`NE` of strings
  run as native C: `same("hi", "hi")` exits 1, `same("hi", "no")`
  exits 0, and `diff("hi", "no")` exits 1. Array equality stays
  refused. `at` as `ARR_GET` and `str_length` as `STR_LEN` run as
  native C: `via_at` exits 7 and `slen("hi")` exits 2.
  `STR_SUBSTR` runs as native C: `slice("hi")` equals `"h"`.
  Substring of arrays and `STR_TRIM` stay refused. `ARR_NEW` of
  `array<int>` runs as native C: `blank_l` exits 0. Void
  `list_int_push` (`ARR_PUSH` then `POP`) mutates the local:
  `grow_l` exits 7. `STR_CHAR_AT` runs as native C: `ch` exits
  104 and an out-of-range index is `-1`. Void `list_string_push`
  (`ARR_PUSH` then `POP`) mutates the local: `blank_s` exits 0 and
  `grow_s` length is 2. I classify `ARR_NEW 1` as a string list from
  the pushed value. Mixed int/string records run as native C: `get_s`
  length is 2. Nested record fields stay refused. Lists of records
  run as native C: `blank_t` exits 0 and `grow_t` length is 2. I
  classify `ARR_NEW 1` as a record list from the pushed value.
  Four-field `LexerToken` records and `List<LexerToken>` run as native
  C: `get_v` and `grow_lex` length are 2. `STR_STARTS_WITH` and
  `STR_ENDS_WITH` run as native C: `has_pre("hi")` and `has_suf("hi")`
  exit 1, and `"no"` exits 0. `STR_SPLIT` stays refused. `ARR_SET` of
  `array<int>` runs as native C: `put_l` exits 9. `ARR_SET` of a
  record list runs as native C: `put_t` length is 2. `ARR_SET` of a
  string list runs as native C: `put_s` length is 2. Generic `LT`
  from `for` runs as native C: `upto` exits 6. `CALL` of `void`
  runs as native C: `via_quiet` exits 0. `CALL` of a pin record
  runs as native C: `via_o` exits 1. `CALL` of a mixed int/string
  record runs as native C: `via_tok` exits 2. `CALL` of `array<int>`
  runs as native C: `via_ones` exits 2. `CALL` of `List<int>`
  runs as native C: `via_new_l` exits 0. `CALL` of a non-empty
  `List<Tok>` runs as native C: `via_one_t` exits 2. Nested records
  stay refused. Empty `List<Tok>` `CALL` (`ARR_NEW 1` then `RET`,
  no `ARR_PUSH`) staying `narr_t` is a separate pin.
  `CALL` of a non-empty `List<LexerToken>` runs as native C:
  `via_one_lex` exits 2. `I64_GE_S`/`I64_LE_S` run as native C:
  `via_az` exits 1. `ENUM_VAL` runs as native C: `via_tag` exits
  the discriminant 19. Imported `ENUM_VAL` runs as native C:
  `via_tok_mod` exits 19. `via_imp_add` exits 42. `via_raw` exits 7.
  `CAST_INT` of a string runs as native C: `via_parse_n` exits 7.
  Casting arrays stays refused. `make test-nvm2c` (449 passed).

## [4.5.0] - 2026-09-07

### Added
- 4.1–4.5 public-release bar (docs, runtime, leftover 4.1) (#247)
- 4.1–4.4 Forth evidence, catalogs, NSI, and POSIX fabric
- close interpreter example failures and enable nanoc --bench
- add a lit OpenGL atelier of high-polygon GLUT solids
- replace the GPU ocean window with an animated Julia set
- ship an SDL editor with a persistent NanoLang session
- close Forth kernel defects and remaining Core words
- compile Forth colon words to verified NanoISA
- add Forth dictionary headers and nested input sources
- add a Forth session runtime on NanoISA

### Changed
- record 4.4 product merge and coverage skip on main (#246)
- fan out the Forth 2012 Core suite work
- record the OpenGL atelier as an addition, not a change
- pin Forth 2012 suites, licensing, and Gforth differentials
- Refresh documentation before tagging, not after (#232)
- Document the verification rules 4.0 added (#231)
- Give the deck a thesis and cover the hardening it omitted (#230)
- Publish the 4.0 deck and narrative with link sharing (#229)
- regenerate the developer deck for 4.0 (#228)

### Fixed
- Linux shared objects and `libnano_session.so` now compile with `-fPIC` so
  TLS in `transpiler.o` links on Ubuntu. i18n user-guide drafts keep
  English relative links to `generated/*.md`; the markdown link checker
  accepts `userguide/generated/` as the target instead of requiring copies
  under each language directory.

### Added
- 4.5 effects-to-policy, trap journal, and observability library
  (`src/nsi_policy.c`, `src/nsi_journal.c`, `src/nsi_obs.c`,
  `schema/nsi/effect_map.v0.json`, `docs/NSI_EFFECTS.md`). Inventory and
  deployment manifest from declared effects; reject uncovered grants.
  Journal record/replay/mock/fault with SHA-256 and HMAC-SHA256. Trace
  and provenance across NanoVM, router, service, and host. The journal
  is a tested library, not a hook on every `vm.c` trap.
  `make test-nsi-policy test-nsi-journal test-nsi-obs`.
- Release tooling accepts an explicit `X.Y.Z` version so product phases
  4.1–4.5 can ship as `v4.5.0` instead of five minor bumps from `v4.0.0`
  (`./scripts/release.sh 4.5.0`). `make nanoc` aliases `bin/nanoc` for
  the CI bench job.
- NSI v0: stable interface, method, type, error, capability, and
  parameter ids (`schema/nsi/examples/log.nsi.json`, `src/nsi.c`,
  `docs/NSI.md`). Parameter direction, ownership, lifetime, mutability,
  optionality, and streaming fail closed. Typed payloads (record, variant,
  array, string, binary, resource, callback, async) and versioned errors
  are in the same loader. `nl_nsi_compat` answers whether a client of an
  older document can call a newer one. Omitted `params`, omitted `kind`,
  `opaque`, and unknown keys such as `c_type` fail closed.
  `src/nsi_gen.c` emits NanoLang, Forth, Python, Rust, and C++ stubs,
  dispatch, frames, validation, docs, mocks, compatibility comments, and
  NanoISA imports (`make test-nsi-gen`). Module manifests carry a portable
  `nsi` block; `module.json` stays build metadata
  (`make test-nsi-manifest`, `schema/nsi/inventory.json`).
  `src/nsi_runtime.c` invokes by method id over in-process, mock, and
  local-process adapters with hello, queues, idempotence, auth, and typed
  handles (`make test-nsi-runtime`, `docs/NSI_TCB.md`). `make test-nsi`.
- Isolated SDL editor walker (`bin/nano_emacs_worker`): length-prefixed
  pipe protocol, crash-restart that keeps buffers, freeze-defun via
  `nano_vm` grandchild (`docs/NANO_EMACS.md`, `make test-nano-emacs-worker`).
  The frame does not `dlopen` the interpreter. I do not claim GNU Emacs
  compatibility.
- Capability runtime and POSIX service fabric (4.4): unforgeable `NlCap`
  tokens, rights attenuation, transfer, revocation, generation bump
  (`src/nsi_cap.c`, `make test-nsi-cap`). Capability-scoped shared
  memory with copy fallback and payload-size benches (`src/nsi_shm.c`,
  `make test-nsi-shm`). Supervisor, `NlHost` POSIX/inproc adapters,
  scoped log/fs/process/net/audio/graphics/gpu/python services, remote
  cap denial, and the SDL editor as a fabric client of `editor.walker`
  / `editor.freeze` (`src/nsi_fabric.c`, `docs/NSI_FABRIC.md`,
  `make test-nsi-fabric`). I do not claim a kernel, GNU Emacs
  compatibility, or a CUDA/CPython wrap.
- UTF-8 message catalogs for `en`, `zh`, `hi`, `es`, `ar`, and `fr`
  (`catalogs/messages/`, `src/catalog.c`). Human stderr looks up the
  process locale; JSON/TOON stay English. `make test-catalog` and
  `make test-locale-catalog` (CIO01 for six languages, L0003 for zh).
  Logs carry event ids (`LOG01`–`LOG04`) and drop bidi overrides and ANSI
  (`make test-log-utf8`). Lexer unexpected bytes are `L0003`; the
  self-hosted lexer fails closed to match C. `nanoc` accepts flags before
  the input file. User-guide builder emits six editions with
  `lang`/`dir`/`hreflang`, translated navigation titles, translation
  memory, and machine drafts under `userguide/i18n/`. RTL editions isolate
  code fences LTR so braces stay source order. I do not call the
  system internationalized.
- UTF-8 at JSON/TOON diagnostic emit (invalid fields become
  `<invalid UTF-8>`), `module.json` fail closed, and Markdown docgen
  source/module name fail closed. Identifiers are ASCII
  `[A-Za-z_][A-Za-z0-9_]*`; non-ASCII fails closed (`L0003`). Typechecker
  `emit_context_error` titles are unique `E001`–`E034`. Unicode FFI:
  grapheme, NFC/NFD, casefold, display width (`make test-unicode-ffi`).
  Logs, catalogs, and translated guides landed in later Unreleased items.
  I do not call the system internationalized.
- Strict UTF-8 of `.nano` source at compile (`CSRC01`, `src/utf8.c`). I
  reject overlong encodings, surrogates, truncated sequences, and code
  points above U+10FFFF. `nl_string` and `bstr_validate_utf8` share the
  walker. User-program `cc` links `src/utf8.c`. JSON/TOON emit, `module.json`,
  and Markdown docgen also require UTF-8.
- Pipeline compiler diagnostic IDs (`src/diag_id.c`, `diag_en` in
  `src_nano/compiler/diagnostics.nano`). English is a lookup. Typechecker
  `emit_context_error` titles are unique `E001`–`E034`.
- 5.0 compilation contract: I emit one verified `.nvm`; C11/LLVM/Wasm/GPU
  are translators; bootstrap compares `.nvm`; native AOT does not embed
  `nano_vm` (`docs/NANOISA_ONLY.md`, Phase 20). I do not execute that
  rewrite in 4.x. The former 5.0 operating-environment theme is 6.0.
- Process locale: `nanoc --locale <tag>` and `--print-locale`, with
  `NANO_LOCALE` then POSIX `LC_ALL`/`LANG` then `en` (`src/locale.c`).
  Invalid CLI/`NANO_LOCALE` fail closed. Diagnostics still render English.
- Forth 2012 Core Ext words `VALUE`/`TO`, `MARKER`, `CASE`, `PARSE-NAME`,
  `BUFFER:`, `DEFER`/`IS`/`ACTION-OF`, `HOLDS`, `S\"`, `C"`, `.R`/`U.R`,
  `UNUSED`, `SAVE-INPUT`/`RESTORE-INPUT`, `REFILL`, and `SOURCE-ID` on the
  NanoISA session. Jackson `coreexttest.fth` passes under
  `make test-forth-coreext`. Passing a suite is evidence; I still do not
  claim Core Ext as a banner.
- Jackson Double Number words (`D+`, `D-`, `D2*`, `D2/`, `M*/`, `2CONSTANT`,
  `2VALUE`/`TO`, trailing-`.` double literals, and the rest of Jackson's
  Double list) on the NanoISA session. Jackson `doubletest.fth` passes under
  `make test-forth-double`. Passing a suite is evidence; I still do not
  claim Double as a banner.
- Jackson String words (`-TRAILING`, `/STRING`, `BLANK`, `CMOVE`, `CMOVE>`,
  `COMPARE`, `SEARCH`, `SLITERAL`, `UNESCAPE`, `REPLACES`, `SUBSTITUTE`) on
  the NanoISA session. Jackson `stringtest.fth` passes under
  `make test-forth-string`. Passing a suite is evidence; I still do not
  claim String as a banner.
- Jackson Search Order words (`WORDLIST`, `GET-ORDER`/`SET-ORDER`,
  `FORTH-WORDLIST`, `ALSO`/`ONLY`/`PREVIOUS`/`FORTH`/`DEFINITIONS`,
  `SEARCH-WORDLIST`) on the NanoISA session. Jackson `searchordertest.fth`
  passes under `make test-forth-searchorder`. Passing a suite is evidence;
  I still do not claim Search Order as a banner.
- Jackson File Access words (`CREATE-FILE`, `OPEN-FILE`, `READ-LINE`,
  `INCLUDED`/`INCLUDE`/`REQUIRED`/`REQUIRE`, and the rest of Jackson's File
  list) on the NanoISA session. Jackson `filetest.fth` passes under
  `make test-forth-file`. Passing a suite is evidence; I still do not
  claim File Access as a banner.
- Jackson Memory-Allocation words (`ALLOCATE`, `FREE`, `RESIZE`) on the
  NanoISA session. Heap allocations do not move `HERE`. Jackson
  `memorytest.fth` passes under `make test-forth-memory`. Passing a suite
  is evidence; I still do not claim Memory-Allocation as a banner.
- Jackson Locals words (`{:`, `(LOCAL)`, `TO` of a local) on the NanoISA
  session. Locals are per-call NanoISA slots, so they are recursive and
  reentrant. Jackson `localstest.fth` passes under `make test-forth-locals`.
  Passing a suite is evidence; I still do not claim Locals as a banner.
- Jackson Facility structure words (`BEGIN-STRUCTURE`, `END-STRUCTURE`,
  `+FIELD`, `FIELD:`, `CFIELD:`) on the NanoISA session. Jackson
  `facilitytest.fth` passes under `make test-forth-facility`. Passing a
  suite is evidence; I still do not claim Facility as a banner.
- Jackson Programming Tools words (`AHEAD`, `[IF]`/`[ELSE]`/`[THEN]`,
  `CS-PICK`/`CS-ROLL`, `[DEFINED]`/`[UNDEFINED]`, `N>R`/`NR>`, `SYNONYM`,
  `TRAVERSE-WORDLIST`, `NAME>COMPILE`/`NAME>INTERPRET`/`NAME>STRING`) on
  the NanoISA session. Jackson `toolstest.fth` passes under
  `make test-forth-tools`. Passing a suite is evidence; I still do not
  claim Programming Tools as a banner.
- Jackson Floating-Point words (IEEE binary64 stack, `D>F`/`F>D`,
  `FLITERAL`, `>FLOAT`, `F~`, `REPRESENT`, and the Jackson `ak-fp-test.fth`
  arithmetic/trig/exp set) on the NanoISA session. Jackson `ak-fp-test.fth`
  passes under `make test-forth-float`. Passing a suite is evidence; I still
  do not claim Floating-Point as a banner.
- UTF-8 Extended Character words (`CHAR`/`[CHAR]` decode a code point,
  `XC@+`, `X-SIZE`, `+X/STRING`, `XC-SIZE`, and the rest of the Xchar hosts)
  on the NanoISA session. Jackson has no xchar file. Session tests under
  `make test-forth-session` are the evidence. I still do not claim Extended
  Character as a banner.
- Block and Block Ext words (`BLOCK`/`BUFFER`/`UPDATE`/`FLUSH`/`LOAD`/`LIST`/`THRU`,
  block `REFILL`/`SAVE-INPUT`, and `\` to the next 64-character line) on a
  disposable 32-block RAM image. Jackson `blocktest.fth` passes under
  `make test-forth-block`. Passing a suite is evidence; I still do not claim
  Block as a banner.
- `vm_sync_new_functions` appends decode/dispatch for newly published NanoISA
  functions without freeing a running caller's decoded instructions, so a
  colon defined inside `EVALUATE` is invokable (Jackson `SSQ7`/`SSQ9`).
- pin Forth 2012 revisions, licensing, and Gforth 0.7.3 pi.fs differentials
- own one NvmModule and persistent VmState per Forth session, with Forth stacks, virtual addresses, and file handles
- keep Forth dictionary headers, word lists, and nested input sources on the session
- compile colon definitions privately to verified NanoISA, publish atomically, and bind `OP_CALL` / `RECURSE` to reserved execution tokens
- compile Forth `IF`/`ELSE`/`THEN`, `BEGIN`/`UNTIL`/`AGAIN`, and `WHILE`/`REPEAT` with a checked control stack
- restore Forth stacks and input sources on `CATCH`/`THROW`; dictionary words
  `CATCH` and `THROW` nest invoke and HALT the outer NanoISA function
- `BYE` is a Forth word; the NanoISA REPL does not special-case the line `bye`
- Forth 2012 Core words `>BODY`, `>NUMBER`, `POSTPONE`, `ABORT"`, `KEY`,
  `ACCEPT`, and `QUIT` on the NanoISA session (`COMPILE,` exists because
  `POSTPONE` of a non-immediate word compiles it)
- add `bin/nano_forth` as the NanoISA session REPL and copy it to `bin/forth` so `sdl_forth_ide` always has its PTY child
- ship a windowed SDL editor (`examples/emacs/nano_emacs.nano`) that evaluates NanoLang in a persistent tree-walker session
- add an OpenGL atelier of high-polygon GLUT solids under four moving lights
- ship `examples/language/data/words.txt` and a built-in fallback so `nl_random_sentence.nano` does not need `/usr/share/dict/words`
- add `examples/bench_sample.nano` with two zero-parameter `bench_*` workloads
- `make test-forth-examples` runs 280 T{ cases from `examples/language/forth/`
  through Jackson `tester.fr`; `make test-forth-ide-smoke` compiles
  `sdl_forth_ide` when SDL2 is present
- `tests/forth/forth2012_skips.txt` records forth200x and Jackson gaps I do
  not treat as banners
- BCP 47 tag parse, fallback chains, and separate language / script / region /
  encoding / collation / direction fields (`src/bcp47.c`, `make test-bcp47`).
- `nvm2c` translates a closed NanoISA subset (i64 arithmetic, locals, `CALL`,
  `RET`/`HALT`) to structured C11 that `cc` compiles; `make test-nvm2c` pins
  `add(40, 2)` exiting 42 without embedding `nano_vm`. This is not a product
  CLI and does not replace `transpiler.nano` or `wrapper_gen`.
- `docs/NANOISA_HL_ROUNDTRIP.md` records whether NanoISA metadata is rich
  enough to reconstruct high-level languages; I do not claim that it is.
  Diagnostics and guides still render English. I do not call the system
  internationalized.

### Changed
- split the Forth 2012 Core gate into four parallel suite items (Jackson vendor, forth200x inventory, `make test-forth-jackson`, coverage matrix)
- `examples/language/forth/` T{ cases use Forth 2012 stack pictures (`S>D`
  before `FM/MOD`/`SM/REM`, `CREATE` for comma'd cells, `RECURSE`, `DEFER`
  for mutual recursion)
- replace the GPU ocean window with an animated OpenCL/CUDA Julia set
- `bin/nano` sets `NANO_INTERPRETER=1`; the sieve and Game of Life use a smaller workload under the interpreter
- libdispatch examples print `SKIP:` and exit 0 when GCD is not available, and also under `bin/nano` because the tree-walker cannot run GCD callbacks
- enable the CI `bench` job: `nanoc --bench` measures the tree-walker and fails on a zero ns/op; it does not compare against a stored 2× baseline
- `nvm2c` translates a closed NanoISA subset (i64 arithmetic, locals, `CALL`,
  `RET`/`HALT`) to structured C11 that `cc` compiles; `make test-nvm2c` pins
  `add(40, 2)` exiting 42 without embedding `nano_vm`. This is not a product
  CLI and does not replace `transpiler.nano` or `wrapper_gen`
- record whether NanoISA metadata is rich enough to reconstruct high-level
  languages in `docs/NANOISA_HL_ROUNDTRIP.md`; I do not claim that it is

### Fixed
- `VARIABLE` allots the data cell after the name, matching `CREATE`
- `forth_take_word` consumes the trailing blank so `S"` does not include a
  leading space; interpret `S"` still uses one transient `WORD` buffer
- a runtime host that returns `0` while interpreting fails closed instead of
  re-entering its trampoline
- `EXECUTE` of a compile-only word is rejected
- aborting an unclosed colon sets `STATE` to interpret
- unsigned single-cell number parse accepts `2^64-1` as all-bits-one

## [4.0.0] - 2026-09-03

### Added
- infer operand types through every basic block (#224)
- check ownership effects and frame depth (#221)
- collect reference cycles (#220)
- add tested symbolic assembly examples (#219)
- make canonical disassembly lossless (#215)
- prove return shape statically (#214)
- make the v2 module format the default (#210)
- compute and check max operand depth (#208)
- emit and load v2 modules behind --emit-nvm-v2 (#207)
- convert between NvmModule and the v2 container (#209)
- whole-module v2 serialization and cross-section validation (#205)
- encode and decode the v2 IMPORTS, LINKS, METADATA and DEBUG sections (#204)
- encode and decode the v2 FUNCTIONS and GLOBALS sections (#203)
- encode and decode the v2 LAYOUTS section (#202)
- encode and decode the v2 SIGNATURES section (#201)
- encode and decode the v2 CONSTANTS section (#200)
- add the bounds-checked cursor for v2 section decoding (#199)
- harden function ranges and branch targets (#128)

### Changed
- dispatch through a label table where the compiler allows (#225)
- measure execution instead of process startup (#222)
- pin the wire byte layout and test the large-payload path (#216)
- measure computed-goto dispatch and record the decision against it (#198)
- name the right branch in the plan's Task 13 collision note (#197)
- MAC task task_e65fad401e65d182556b185627ffb07d: Verify linked-module calls and every opcode family rather than selected operands only (#196)
- write the 4.0 release document and bring the changelog up to date (#195)
- Implementation plan for the NanoISA v2 module format (#194)
- Implement the v2 module container: header and section directory (#193)
- cover the extern-call argument limit on the execution path (#192)
- MAC task task_b8c2e917cfa7532a0914fcfde71877d3: Make argument limits consistent across imports, traps, direct FFI, and co-process calls (#169)
- Bundle a generated WAV so sdl_audio_wav has a default (#179)
- cover non-wrapping over-long code ranges and valid import signatures (#173)
- Specify the NanoISA v2 module format (#172)
- scope module signing explicitly to 5.0 (#191)
- clean NanoLang presentation authoring package (#134)
- make the pyyaml install survive PEP 668 runners (#183)
- install pyyaml so schema-check can run (#182)
- record the integration benchmark for the recovered work
- drop local .claude settings swept in by an over-broad git add
- recover: validate complete operand consumption (task_ef9e2d44, approved but never published)
- Make generated-C tracing and profiling dynamically selectable at process startup, with one shared hook mechanism and ... (task_c2cd6757ca772d96fe31bd7982d41333) (#175)
- Replace chained hash-map entries with a measured contiguous implementation (task_dc7b8b55c544fe5a6591d6d1cee27fff) (#155)
- Batch high-frequency host work rather than cross the process boundary for each element (task_23b3c6ceb7c4d6b6fcad219ee27fa97e) (#170)
- Document the portable ISA separately from verified and optimized runtime representations (task_4f706c20951735ab246462ba68853ea2) (#178)
- Use generated typed stubs or a general ABI layer for mixed integer and floating signatures (task_d5b6ebd1146e789c114037cda9fff9d9) (#177)
- Choose one coherent flattened or separately linked module model and test it end to end (task_b25fc1d1d6fcc3ed416974df94f73491) (#166)
- Add an exhaustive coverage check tying every opcode to schema, VM behavior, verifier rules, assembly, disassembly, an... (task_5eaea82338c6eaad7b9ebd2a883e5714) (#165)
- Add malformed-bytecode tests and fuzz the decoder, loader, verifier, assembler, disassembler, and co-process protocol (task_34f40c67d0579535b820f91bb57d00c0) (#164)
- MAC task task_c7dbefc25811eded67074bf807cb3132: Record why every public instruction belongs in the ISA rather than a runtime library
- Make symbolic functions, imports, fields, types, constants, and labels first-class assembler operands (task_de09eed3382c75b6306461634000211f) (#162)
- MAC task task_497f43bf1fabcc1d3ffbe19b6227dbc8: Resolve imports once into typed call descriptors
- Eliminate integer-overflow and overlap gaps in code-range and section validation (task_9a9fb97d491f5ecf38c3ea11e65f92cd) (#160)
- MAC task task_4ffa2d2afab91335d5e5d333ed25b5fe: Remove or justify opcodes that no frontend emits, including no-op GC scopes and duplicated closure/string operations
- MAC task task_4ae13273aed3dd8cce7524500d64bf76: Correct disassembler import annotations, branch operand roles, label construction, and binary-string handling
- Verify call arity, result shape, aggregate counts, local/global/upvalue bounds, type tags, and import signatures (task_8ace2f517a3254eac3744561fb3bfb8f) (#158)
- MAC task task_b9423dba41b3c1cb9c99caa77f8cd860: Rewrite verified operations to unchecked private handlers where the proof permits it
- Add unboxed homogeneous arrays for integer, float, boolean, and byte elements (task_cfc51a97dd37b93a2cee7a1080406a04) (#161)
- Replace linear transient-string interning or stop interning transient values (task_17ccd24fce8b2b6f7bd305ce31e92876) (#157)
- MAC task task_1e7b2d8a0eabe16b338e1abf0fa36f13: Consistently use stored string lengths and preserve embedded zero bytes
- Dynamically size globals from serialized declarations instead of embedding 4,096 values in every VM (task_6cac13465b0a9acafe0d0d845b1afe16) (#153)
- MAC task task_deba0dbe48cf9797aae9aab814e5baca: Preinstantiate module constants so string literals do not allocate and search the intern table on every execution
- Initially evaluate local-field load, local increment, compare-branch, union-tag branch, and tail-call fusions (task_c035d8c695fefe01213666c883d50d7e) (#152)
- MAC task task_fc36a92356c08a07fda74fa773684916: Add profile-selected private superinstructions without exposing frontend bookkeeping as portable opcodes
- MAC task task_b5ad6c4b9baaa8fce18d3c739a266d34: Move trimming, case conversion, splitting, replacement, formatting, parsing, and collection algorithms from the ISA i... (#148)
- MAC task task_2885f0a720489b5d3f83f8e5b0cf6e47: Replace special print, assert, and host operations with typed traps where that improves composition (#147)
- MAC task task_2601fac15e38ad2c0f9d3e8f5f861654: Retain only primitive string and aggregate operations justified by representation or measured cost (#146)
- MAC task task_95a5d10435ec578fed9a3395efe74f35: Give each portable instruction one comprehensible meaning and keep operand forms symmetric (#143)
- MAC task task_0fd9121fa91d7b9208e9ea79d09d843c: Separate compact serialized bytecode, verified instruction IR, and optimized dispatch IR (#144)
- MAC task task_ee63b2c32c5b600e16289566bab6ea3b: Build instruction-boundary maps and resolve branches plus direct and tail calls during instantiation; layouts, consta... (#142)
- MAC task task_c0fe61a767e9e204f823d879d50d1757: Define a clean extended-opcode space without treating an opcode value as an instruction count (#140)
- MAC task task_b7c534c9150ee4bb4b8bca62b37b50dd: Add compact constants, short local forms, and compact general operands without making assembly irregular (#139)
- Resolve linked calls to callable handles (#137)
- Regularize calls around verified signatures (#136)
- Implement NanoISA verifier stack-height propagation (#132)
- add NanoLang developer presentation package (#131)
- use indexed decoded dispatch (#130)
- cover malformed container ranges and debug records (#127)
- MAC task task_647c015a6b4b4e76a3c223a6f920b7a1: 4.0 module format and tools workstream (#126)
- publish NanoLang 3.5 release presentation (#125)

### Fixed
- confine symbol visibility to one file (#226)
- repair stack discipline the verifier now checks (#218)
- fix reference ownership at four leaking sites (#217)
- stop treating an unknown stack effect as a verified one (#213)
- keep generated GPU kernels out of source tree (#190)
- reject a quoted string ending in a backslash (#189)
- keep -rdynamic when LDFLAGS is overridden on the command line (#188)
- align the RC heap base and pad the header so payloads are 8-byte aligned (#187)
- initialize four Environment fields that were never set (#186)
- link the verifier everywhere the assembler is linked
- bound format conversions that GCC rejects under -Werror (#184)
- list the verifier routes #158 added so the coverage check passes (#180)
- give modules/nanoisa the verifier its assembler now needs
- assert the replace result while the VM is still alive
- separate verified assembly from fragment assembly
- list the verifier routes #158 added so the coverage check passes
- encode co-process wire fields as little endian (#129)

Work toward 4.0 (NanoISA v2 and NanoVM v2). See [docs/RELEASE_4.0.md](docs/RELEASE_4.0.md)
for what 4.0 is and what remains.

### Added

#### Portable ISA
- Each portable instruction has one comprehensible meaning, with symmetric operand forms; the schema generator refuses to emit a design that breaks either rule.
- Extended-opcode space defined without treating an opcode value as an instruction count.
- Compact constants, short local forms, and compact general operands, without making assembly irregular.
- Signed and unsigned division, remainder, comparison, shifts, carry, borrow, and wide multiplication primitives required by Forth double cells.
- Indexed `PICK` and `ROLL` alongside the basic stack operations.
- Byte-addressed little-endian memory loads and stores at 8, 16, 32 and 64 bits, with unaligned access explicitly supported.
- Layout-driven `AGG_PACK`, `AGG_GET`, `AGG_SET` and `AGG_TAG` replace language-specific struct, tuple and union lowering.
- Direct function references and heap closures separated into unambiguous value tags and constructors.
- Every public instruction records why it belongs in the ISA rather than a runtime library, as a per-opcode justification.

#### Verifier
- Control-flow verification with instruction-boundary validation: branch targets resolve against a real decoded boundary map.
- Stack-height propagation through every basic block, requiring compatible merge states.
- Call arity, result shape, aggregate counts, local/global/upvalue bounds, type tags and import signatures are verified.
- Verified operations rewrite to unchecked private handlers where the proof permits.
- Malformed-bytecode tests and fuzzing for the decoder, loader, verifier, assembler, disassembler and co-process protocol.

#### Runtime and execution
- Indexed predecoded dispatch, with serialized bytecode, verified instruction IR and optimized dispatch IR kept separate.
- Profile-selected private superinstructions, without exposing frontend bookkeeping as portable opcodes.
- Unboxed homogeneous arrays for integer, float, boolean and byte elements.
- Globals sized dynamically from serialized declarations instead of embedding 4,096 values in every VM.
- Module constants preinstantiated so string literals do not allocate and search the intern table on every execution.
- Chained hash-map entries replaced with a measured contiguous implementation.
- Linear transient-string interning replaced.
- Generated-C tracing and profiling selectable at process startup through one shared hook mechanism, with no per-event environment lookups.

#### FFI and modules
- Mixed integer and floating call signatures dispatch through generated typed stubs (`scripts/gen_ffi_dispatch.py` → `src/nanovm/ffi_dispatch_generated.h`), placing integer and pointer arguments in general-purpose registers and floating-point arguments in FP registers per the platform ABI, without depending on libffi.
- Imports resolve once into typed call descriptors.
- High-frequency host work is batched rather than crossing the process boundary per element.
- Separately linked module model, tested end to end.
- Symbolic functions, imports, fields, types, constants and labels are first-class assembler operands.
- Exhaustive coverage check tying every opcode to schema, VM behaviour, verifier rules, assembly, disassembly and tests.
- NanoISA v2 module container, now the default on-disk format (`NVM\x02`): a 40-byte header with independent format and ISA versions, feature-bit negotiation, and a 64-bit section directory bounded by subtraction so an offset near the top of the range cannot wrap into it. All ten sections encode and decode -- METADATA, CONSTANTS, SIGNATURES, LAYOUTS, FUNCTIONS, CODE, GLOBALS, IMPORTS, LINKS and optional DEBUG. Constants carry explicit lengths, so a string holding an embedded zero survives; signatures are deduplicated, so comparing signature indices is a valid equality test; layouts nest only downward, so the table is acyclic by construction. See `docs/superpowers/specs/2026-09-01-nanoisa-v2-module-format.md`.
- `examples/audio/nanolang-test-tone.wav`, generated by `scripts/generate_test_tone.py`, so the WAV playback example runs without arguments.

### Changed
- `--emit-nvm` writes the v2 container; `--emit-nvm-v2` remains accepted as a retired alias. The loader dispatches on the container byte and refuses a v1 module with `module was built for NanoISA v1 (NVM\x01); rebuild it with nanoc 4.0 or later`. `.nvm` files are build artifacts rather than distributed packages, so the fix is to rebuild instead of carrying a compatibility path that would have to stay correct forever.
- Function entries record a verifier-confirmed `max_stack`. The producer computes the maximum operand depth and the loader confirms it, rejecting a module that declares less depth than it uses; a disagreement between producer and verifier otherwise surfaces as a stack overflow at run time.
- Higher-level string and collection algorithms moved out of the ISA into runtime libraries; only representation- or cost-justified primitives remain.
- Special print, assert and host operations replaced with typed traps.
- Separate-module `CALL_MODULE` calls resolve to callable handles during linking instead of carrying module and function index pairs through dispatch.
- Direct, indirect, tail, imported and linked calls regularized around verified signatures.
- Instruction-boundary maps built and branches, direct calls and tail calls resolved during instantiation.
- Opcodes no frontend emits removed: `GC_SCOPE_ENTER`, `GC_SCOPE_EXIT` and `CLOSURE_CALL`.
- Assembly is verified by default; `asm_assemble_unverified` is available for callers that legitimately assemble a fragment rather than a program.
- Stored string lengths used consistently, so embedded zero bytes are preserved.
- Argument limits are consistent across imports, traps, direct FFI and co-process calls, sharing a single `NANO_MAX_FFI_ARGS`.

### Fixed
- `OP_CALL_EXTERN` no longer silently truncates an over-long argument list. It clamped the declared count and popped only that many, dropping the remaining arguments *and* leaving them on the operand stack, which desynchronized it for every instruction that followed.
- Four `Environment` fields (`trace`, `gpu_target`, `profile_runtime`, `profile_flamegraph_path`) were never initialized. Reading an uninitialized `_Bool` is undefined behaviour a compiler may optimize on, and one of the four is a pointer.
- The reference-counted heap handed out payloads that were only 4-byte aligned: the header was 12 bytes and the heap base had alignment 1. Any object holding a pointer or a double was under-aligned.
- Heap-buffer-overflow in the assembler on a quoted string ending in a backslash; the escape branch consumed the terminator and the loop then read past the end of the line buffer.
- Co-process wire fields encode as little endian.
- Function ranges and branch targets hardened in the verifier; integer-overflow and overlap gaps eliminated in code-range and section validation.
- Disassembler import annotations, branch operand roles, label construction and binary-string handling corrected.
- Build and CI: `pyyaml` installed so `schema-check` can run; `-rdynamic` preserved when `LDFLAGS` is overridden; the verifier linked everywhere the assembler is; format conversions bounded where GCC rejects them under `-Werror`; generated GPU kernels kept out of the source tree.

## [3.5.0] - 2026-08-30

Release presentation: [NanoLang 3.5](docs/RELEASE_3.5.md)
Current project README: [README.md](README.md)
GitHub release: [v3.5.0](https://github.com/jordanhubbard/nanolang/releases/tag/v3.5.0)

### Added
- add runtime-selectable diagnostics (#120)

### Changed
- scope remaining Phase 12 work to 4.0 (#123)
- scope NanoISA v2 completion to 4.0 (#122)
- record verified 3.5 diagnostics work (#121)

## [3.4.14] - 2026-08-30

### Added
- add tail-call support
- make function results explicit
- predecode verifier boundaries
- separate function references and closures
- add integer and memory primitives
- introduce typed scalar operations
- add typed scalar instructions
- complete baseline profiling counters
- establish measured v2 roadmap
- add integer pi example
- support persistent language sessions

### Changed
- require AI-scoped release workflows
- Integrate predecoded NanoVM dispatch (task_133994191eff4025969f236aca8cb6c5) (#117)
- dispatch predecoded instructions
- add multi-language NanoISA laboratory
- map Nano platform releases
- retire direct LLVM and Wasm paths
- regularize aggregate operations
- finish typed arithmetic lowering
- isolate array arithmetic
- generate metadata from schema
- define v2 measurement contract
- remove executable debug metadata

### Fixed
- expose cache rebuild helpers
- rebuild caches after idle mutation
- render default colors safely
- discard array set result

## [3.4.13] - 2026-08-30

### Fixed
- keep Forth IDE interpreter alive (#111)

## [3.4.12] - 2026-08-28

### Added
- turn ocean demo into a compute stress scene

### Changed
- release.sh pushes main directly, so make release always fails at the final step (task_89709a53b78f4b5f83466c2a4fb14013) (#108)

## [3.4.11] - 2026-08-27

### Changed
- stop launching the GLUT examples during test and release runs; `make test-glut-launch` runs them on demand

### Fixed
- reject use-after-consume of a resource instead of reporting it and compiling anyway (#106)
- probe installed apt packages with dpkg-query so the check never stalls in the system pager (#105)
- stop dropping commits from the generated changelog

## [3.4.10] - 2026-08-27

### Fixed
- auto-install SDL dependencies on Linux instead of dying in the C compiler (#102)

## [3.4.9] - 2026-08-26

### Fixed
- stop dpkg from paging Linux module-install probes through a TTY pager
- find Homebrew keg-only readline on macOS and print brew hints instead of apt-get
- keep the Darwin module-validator regression from failing when optional Homebrew formulae are absent

## [3.4.8] - 2026-08-26

### Added
- rebuild GPU Ocean as a two-dimensional CUDA/OpenCL surface with perspective rendering, distributed hull response, and a vessel wake

### Changed
- build the complete native example set before opening the example launcher
- document the GPU runtime and generated-kernel requirements used by GPU Ocean

### Fixed
- launch prebuilt examples without recompiling them synchronously on the SDL event thread
- include launcher-visible GPU artifacts in the normal example build

## [3.4.7] - 2026-08-25

### Fixed
- keep strict example builds warning-free
- remove recursive playground shadow execution
- eliminate example compiler warnings

## [3.4.6] - 2026-08-25

### Added
- first-class Performance Monitoring and LLM Optimization documentation for `-pg`

## [3.4.5] - 2026-08-24

### Added
- document native `-pg` profiling, child-process collection, and evidence-based LLM performance tuning in the published user guide

### Changed
- install binaries under `$HOME/.local` by default while preserving `PREFIX` overrides for installation and removal

### Fixed
- compile SDL NanoAmp examples that use arrays of opaque `Mix_Music` handles

## [3.4.4] - 2026-08-24

### Fixed
- honor NANO_BUILD_CACHE and keep tracker samples in Downloads

## [3.4.3] - 2026-08-24

### Fixed
- gate strict example builds in CI
- retire stale runtime claims (#85)
- publish release dependency metadata (#84)
- make scalar WASM programs executable (#82)

## [3.4.2] - 2026-08-23

### Changed
- consolidate staged REPL, Forth, logging, HashMap, prime, SDL_image, and large
  project examples around one maintained implementation per lesson
- replace placeholder tracing, texture, file pipeline, SQLite, libevent, libuv,
  sprite animation, and verified-boundary demonstrations with working examples

### Fixed
- restore visible GPU Ocean output on macOS with generated OpenCL kernels, a
  stable kernel path, renderer fallback, and explicit SDL diagnostics
- restore the Code Display Widget demo with portable font lookup, renderer
  fallback, and initialization diagnostics
- repair Checkers multi-jumps, Asteroids wave resets, rotated Bullet geometry,
  raytracer quit controls, NanoAmp track ownership, and GPU Ocean fallback

## [3.4.1] - 2026-08-23

### Fixed
- stop complete launcher process groups, clean up examples on exit, and provide
  tailored icons for every launcher-visible graphical example
- generate and load the `matmul` PTX and OpenCL kernels from stable repository
  paths so the example runs through the launcher on OpenCL systems

## [3.4.0] - 2026-08-22

### Added
- make nanoisa_print assembler-canonical
- add NanoISA I/O facade and dump CLI

### Fixed
- snap carried ball to gripper site in robot sorter example
- repair MuJoCo regressions and harden module cache invalidation
- GNU Make treats # as comment inside \$(shell ...) — remove #include probes

### Fixed
- initialize GLUT once through `modules/glut` so the GLFW-owned OpenGL examples
  (`opengl_solar_system`, `opengl_teapot`) stop aborting with "glutSolidSphere
  called without first calling glutInit"
- scope `--profile-runtime` to the backends that implement it. The flamegraph
  counters and the `.nano.prof` writer are injected into the C the transpiler
  emits, so only the native backend can produce a profile. `--target
  wasm|ptx|opencl|c|riscv` and `--llvm` used to accept the flag and silently
  emit an unprofiled artifact; they now fail with an explicit
  unsupported-backend diagnostic. `--help` and the 3.3.x entry for #51 are
  qualified to match.

### Removed
- `src/wasm_profiler.{c,h}`, a stub whose header claimed the real
  implementation "lives in feat/wasm-runtime-profiler (PR #51, merged)" and was
  lost in the merge. PR #51 touched only `Makefile.gnu`, `src/main.c`,
  `src/nanolang.h`, `src/stdlib_runtime.{c,h}`, `src/transpiler.c` and
  `tests/unit/test_profile_runtime.nano` — no WASM profiler source ever existed
  to restore. `wasm_profiler_create()` returned `NULL` and nothing called it.
  `tests/test_wasm_profiler.c` is now a real test of the backend support policy.

## [3.3.8] - 2026-05-30

### Added
- restrict catalog to graphical, non-trivial demos
- api_lessons coverage + Makefile wiring + bullet/handle fix
- honor install_command/test_command from packages.json
- add nanoc add command and persist dependencies on save
- enhance disassembly and add debug-info stripping

### Fixed
- wire auto-install + framework-cask runtime lookup
- nl-1f3 — keep stderr off the JSON pipe in bd list --json
- nl-phi — track current block label, fix phi predecessor after return
- nl-d7n — remove dead nano_* declares, fix lli compatibility
- nl-cuc — restore is_union_construct guard
- nl-3du — replace silent i64.const-0 stubs with hard errors
- restore generated C compilation
- transpiler header resolution, fs/process linkage, beads.nano stdlib completions
- Backend Matrix — C/LLVM/WASM/parser failures across 5 test programs
- add pure keyword support to self-hosted lexer and parser

## [3.3.7] - 2026-04-20

### Added
- shared-memory mailbox fast path for nano_cop FFI
- apply pure fn across codebase; fix BUILTIN_PURE and module import propagation
- phase 2+3 — pure extern fn and auto-TCO for pure functions
- add 'pure fn' annotation with compile-time purity enforcement
- add OpenCL C backend — --target opencl + transparent CPU fallback
- add ocean.nano — GPU ocean simulation with Archimedes ship physics
- add matmul.nano — 2D GPU operations demo using gpu_launch2d
- gpu_load/gpu_store PTX intrinsics + saxpy/reduce_sum/vector_scale examples
- full CUDA GPU support — PTX builtins, runtime module, and example
- REPL persistent history and session save/load

### Fixed
- correct upvalue index validation and add struct/enum/union def_idx bounds checking
- transpiler variable shadowing, SDL quit, profiling CWD, and process capture
- handle zero-capacity ModuleList in module_list_add
- initialize env->effect_registry = NULL in create_environment()
- resolve both CI failures — ASAN stack-use-after-return + add GPU intrinsic tests
- add gpu_launch5 for 5-arg kernels; fix saxpy missing n arg
- correct DynArray layout in cuda_runtime.c + atom.global.add.u64
- use atom.global.add.u64 for gpu_atomic_add
- C forward decls for GPU intrinsics + shadow tests for gpu.nano
- use u32 register pool (%r) for special register reads
- replace em-dash with colon in PTX header comment
- fix -Werror=type-limits for uint32_t >= 0 comparison
- increase code buffer size to prevent overflow in test_add_array_array
- resolve isatty type clash in nl_forth_interpreter

## [3.3.6] - 2026-04-02

### Added
- source-mapped stack trace tests + multi-frame coverage

### Fixed
- sdl_forth_ide native binary path + nl_forth_interpreter TTY REPL

## [3.3.5] - 2026-04-02

### Added
- forth REPL + release sync fix
- add FIG-Forth interpreter example (token-threaded, ~2500 lines)
- generic function monomorphization (type-variable T in fn signatures)
- --doc-md flag — GFM Markdown doc export from triple-slash comments
- CodeMirror 6 editor, share permalink, AgentFS hosting
- nanolang interactive playground (browser editor + eval server, port 8792)
- complete installable .vsix — semantic tokens, format-on-save, tasks, packaging
- VS Code extension for nanolang (LSP, syntax highlighting, format-on-save)
- hot-reload — :load, :save, :reload commands
- REPL hot-reload (:reload <module>) — live module reloading without restart

### Changed
- make release script non-interactive by default

### Fixed
- effects tests — mut variable and missing shadow stubs

## [3.3.4] - 2026-04-02

### Added
- add FIG-Forth interpreter example (token-threaded, ~2500 lines)
- generic function monomorphization (type-variable T in fn signatures)
- or-patterns in match, stdlib docs site, match completeness tests
- DWARF debug info emission (--debug/-g flag for LLVM IR and RISC-V backends)
- nanolang benchmark suite — optimizer comparison + regression tracking
- REPL persistent history and session save/load
- WASM SIMD128 auto-vectorization — emit v128 opcodes for numeric patterns
- WASM SIMD128 auto-vectorization for numeric array loops
- property-based test oracle (@property, --proptest, QuickCheck-style shrinking)
- property-based test oracle (@property, --proptest, QuickCheck-style shrinking)
- REPL scripting — nano --script <file> and nano -e '<expr>' modes
- RISC-V assembly backend (riscv_backend.c)
- nano-docs static site generator — make docs builds HTML from userguide/**/*.md
- nano-bench micro-benchmark harness (--bench mode)
- nano-docs — documentation search CLI for nanolang
- typechecker generics — per-call type variable unification with consistency checking
- nano-fmt — canonical code formatter with LSP textDocument/formatting support
- agentOS typed shared-memory ringbuffer library (ringbuf.h)
- nano-to-C transpiler backend (--target c)
- nano-to-C transpiler — nanoc --target c emits clean C99 for seL4 PD embedding
- nano-to-C transpiler backend (c_backend.c)
- standard library — Option, Result, List, Map, Set, Iterator, String
- LLVM IR backend — nanoc --llvm emits .ll for ARM64/x86-64 native code
- LSP semantic tokens (textDocument/semanticTokens/full)
- LSP hover types show row-polymorphic records and type-scheme generalization
- hover displays row-polymorphic HM types
- nano package registry — HTTP server, nanoc pkg CLI, semver resolution, lockfile
- nano package registry + nanoc install/publish subcommands
- generational GC — tri-color mark-sweep cycle collection (Bacon-Rajan)
- PGO (profile-guided inlining) pass — nanoc --pgo <file>.nano.prof
- nanodoc — mdoc-style HTML doc generator for .nano modules
- cooperative scheduler runtime for async/await
- async/await syntax — CPS transform pass, cooperative async functions
- WASM reference-counting GC (refcount_gc.h) + transpiler integration
- algebraic effect system — effect declarations, handlers, effect polymorphism
- coroutine runtime — cooperative scheduler for async/await continuations
- async/await syntax — CPS transform pass, cooperative async functions
- row-polymorphic records — fix build compatibility with main branch
- PTX backend — nanoc --target ptx emits NVIDIA PTX assembly for gpu fn kernels
- PTX backend — nanoc --target ptx emits NVIDIA PTX assembly for gpu fn kernels (#50)
- PTX backend — nanoc --target ptx emits NVIDIA PTX assembly for gpu fn kernels
- native runtime profiler — --profile-runtime emits flamegraph collapsed-stack .nano.prof
- native runtime profiler — --profile-runtime emits flamegraph collapsed-stack .nano.prof (#51)
  (native backend only; see Unreleased → Fixed. #51 was titled "WASM runtime
  profiler" but shipped no WASM support.)
- row-polymorphic records — row var unification, spread, open patterns
- f-string string interpolation — comprehensive tests and userguide docs (#49)
- coverage tracking — --coverage flag, 80% CI threshold, coverage-check target (#48)
- add format/string interpolation builtin to stdlib (closes wq-API-1774937552439)
- coroutine runtime — cooperative scheduler for async/await continuations
- async/await syntax — CPS transform pass, cooperative async functions
- algebraic effect system — effect declarations, handle expressions, type checking
- add package manager (nanoc-pkg) with lockfile support
- structured output formatter — JUnit XML and TAP for CI
- NanoLang interpreter compiled to WASM via Emscripten
- WASM source maps — .wasm.map JSON + sourceMappingURL custom section
- add shadow tests for missing functions; fix stdlib redefinitions
- add WASM binary emit backend (--target wasm)
- DAP-compatible debugger server (breakpoints, step, inspect)
- nanolang Language Server Protocol server (hover, definition, completion, diagnostics)
- add --profile-output flag for structured benchmark JSON
- implement par { } blocks
- improve compiler error messages with structured Note/Hint format
- add guard clauses to match expressions (#20)
- complete stdlib expansion — 7 new builtins + bootstrap rename (#23)
- update extension to v0.2.0 for NanoLang v3.3
- implement module system – circular import detection, private visibility, shadow tests
- add 8 new string stdlib builtins (#23)
- implement ? error propagation operator in C compiler

### Changed
- make release script non-interactive by default

### Fixed
- `generate_effect_perform_stubs()` in transpiler.c: NULL deref on `op_param_types` — falls back to `op_params[j][0].type`
- Free type variable `T` in `get_prefixed_type_name()` now maps to `void*` instead of `nl_T`
- `test_effects.nano` infinite loop: `let i = (+ i 1)` inside while shadowed outer i; fixed with `let mut` + `set`
- `test_generics.nano`: rewrote to exercise real T-parameterized monomorphization; removed undefined symbols
- Stale `modules/std/json/.build/source_hashes.json` cache invalidated to pick up `nl_json_as_float`
- effects tests — mut variable and missing shadow stubs
- remove duplicate AST_EFFECT_OP cases and guard unfinished SIMD block
- resolve CI compile errors from overnight branch merges
- remove stale validate-topology dependency from bench job
- merge-branches.sh use git branch -r for existence check
- resolve main build failures — missing sources, riscv bug, tidy/format collision
- wire algebraic effects syntax — comma parsing, handle forms, perform keyword
- wire full coroutine scheduler for PR #47 rebase
- transpile AST_ASYNC_FN and AST_AWAIT nodes
- add libssl-dev to all apt jobs + -lcrypto to sanitizer/coverage LDFLAGS
- resolve CI test failures — missing test sources + f-string float formatting
- resolve merge conflicts and restore real pass implementations
- row-poly spread syntax, typechecker usage tracking, test_optimizer stub
- cross-platform random seed in sign.c for macOS/FreeBSD
- remove invalid sprint2 test; add macOS OpenSSL paths for sign.c
- add -lcrypto to sanitizer and coverage LDFLAGS for OpenSSL sign.c
- float_to_string preserves decimal point for whole-number floats
- restore missing source files and fix build for algebraic effects integration
- add fn main stubs to ug_fstring_basic and ug_fstring_exprs snippets
- move nl-snippet markers outside [?2004h)0[1;24r[m(B[4l[?7h[?25l[H[J[22;35H[0;7m(B[ New File ][m(B[?25h[24;1H
[?2004lnano fences instead of before them. The userguide_snippets_check scanner does not track fence state, so it found these markers inside fenced blocks and then hit the bare nano and <!--nl-snippet ...--> lines so the marker precedes the fence, as required by the checker.
- add AST_ASYNC_FN/AST_AWAIT cases to type_infer.c
- cross-platform random seed in sign.c for macOS/FreeBSD
- transpile AST_ASYNC_FN and AST_AWAIT nodes
- add TOKEN_ASYNC/AWAIT and PNODE_ASYNC_FN to schema so they survive schema regeneration
- tco_pass.c — suppress unused-function warning on make_return helper
- add stub sources and macOS OpenSSL path so main builds
- build on Linux/GCC — strncpy/strncat warnings and NULL guards
- add checks:write permission and continue-on-error for test reporter
- rename 'handle' in pybridge test; add main to stdlib strings test
- rename 'handle' identifier to avoid collision with new keyword
- cross-platform random seed in sign.c for macOS/FreeBSD
- transpile AST_ASYNC_FN and AST_AWAIT nodes
- regenerate compiler schema — add TOKEN_BAR (shifts EFFECT/HANDLE/WITH/RESUME by 1)
- remove test_str_split from main() accumulator — returns array_length (3), not pass/fail count
- rename local str_index_of_from to avoid conflict with new builtin (#46)
- unblock CI after stdlib expansion (#43)
- forward-declare g_typecheck_error_count before first use in typechecker.c
- add TOKEN_ASYNC/AWAIT and PNODE_ASYNC_FN to schema so they survive schema regeneration
- remove string-returning functions from main() accumulator in test_stdlib_strings
- add missing comma after AST_PAR_BLOCK in ASTNodeType enum
- resolve CI breakages blocking PR merges (#31)
- tco_pass.c — suppress unused-function warning on make_return helper
- remove local str_trim/str_starts_with from vars_repl.nano
- fix examples-full build failures on all platforms
- remove redundant str_trim from simple_repl.nano
- add _POSIX_C_SOURCE=200809L for module compilation on Linux/FreeBSD
- add non-Apple sequential fallback in nano_dispatch.h
- guard Blocks/GCD code with #ifdef __APPLE__, add stubs for Linux/FreeBSD
- initialize env->profile in create_environment(); add platform-specific cflags support
- correct smoke test assertions to match actual REPL output
- use public fs.read/write/exists instead of private file_* variants (#38)
- use GITHUB_REPOSITORY env var instead of context.repo for PR comments
- remove userguide local redefinitions of stdlib builtins (str_starts_with, str_ends_with, str_index_of)
- repair test_stdlib_strings.nano — dedupe fn main()
- remove remaining local redefinitions of str_starts_with/str_ends_with builtins
- resolve two CI breakages on main
- resolve two CI breakages blocking PR merges (#31)

## [3.3.4] - 2026-03-30

### Added
- WebAssembly binary emit backend: `./bin/nanoc program.nano --target wasm -o program.wasm`
  supports int/float/bool types, arithmetic, comparisons, function calls, if/else, and recursion
- DAP-compatible debugger server (`bin/nanolang-dap`, `make dap`): breakpoints, step/next/stepIn/stepOut,
  variable inspection, and stack traces over JSON-RPC stdio — compatible with VS Code and any DAP client
- VS Code extension updated with DAP debug launch configuration
- Shadow tests added for all previously-untested functions across
  nl_forth_interpreter.nano, transpiler.nano, module_loader.nano, and nanoc_v06.nano
- Enable previously-disabled shadow tests for nl_path_dirname and parse_import_path_from_line in nanoc_v06.nano

### Fixed
- Remove local `str_trim` definition in example_discovery.nano (stdlib builtin conflict)
- Remove local `str_starts_with` definition in nl_forth_interpreter.nano (stdlib builtin conflict)

## [3.3.3] - 2026-03-28

### Added
- Fill in all 27 placeholder user guide pages in Part 3 (text processing,
  data formats, web/networking, graphics, OpenGL, game dev, terminal UI,
  testing, configuration) with full API references, examples, and best practices
- Fill in 3 previously-empty API reference pages (StringBuilder, coverage,
  vector2d) by marking their public functions as pub fn

### Fixed
- Reconcile user guide and API doc generator with v3.3.2 stdlib refactor:
  update stringbuilder.md to new sb_* API, update regex.md import paths,
  fix generate_all_api_docs.sh module paths
- Fix pre-existing StringBuilder_to_string shadow test (discarded append
  return value under immutable semantics)
- Fix userguide_build_html HashMap: module pub fn unreachable from regular
  functions; replace with inline get_theme_color() lookup

## [3.3.2] - 2026-03-28

### Changed
- remove redundant stdlib/std, std/json, and unused stdlib files
- retire stdlib/regex.nano in favour of std/regex/regex.nano
- modernize all test files to current nanolang syntax

### Fixed
- mark public APIs as pub fn so API reference generator picks them up
- resolve three CI failures introduced by modernize-tests commit
- restore explicit type annotations for array_pop calls in test_dynamic_arrays.nano
- resolve stage3 bootstrap blockers
- make libdispatch-dev install optional in Concurrency CI job
- add out_path null guard in ffi_loader_find_library
- use memcpy instead of snprintf in nl_walkdir_rec to avoid format-truncation
- suppress GCC format-truncation false positive in eval_io.c
- typechecker null-guard and regex test private API usage
- update .nano files to use public fs.nano API
- correct CI failures — typechecker exit code and module.c truncation
- resolve module paths correctly when cwd is not repo root

## [3.3.1] - 2026-03-27

### Added
- @pure/@associative annotations, frozen let, par blocks — C compiler + .nano sources
- f-string interpolation with automatic type conversion
- Option A dispatch wrapping for module-scope let mut primitives

### Fixed
- array_remove_at return type void→array in builtins registry
- resolve -Werror build failures (fread/system/fgets unused return, strncpy truncation)
- resolve all CI failures
- add explicit Makefile rule for nl_forth_interpreter_vm

## [3.3.1] - 2026-03-27

### Added
- @pure/@associative annotations, frozen let, par blocks — C compiler + .nano sources
- f-string interpolation with automatic type conversion
- Option A dispatch wrapping for module-scope let mut primitives

### Fixed
- resolve -Werror build failures (fread/system/fgets unused return, strncpy truncation)
- resolve all CI failures
- add explicit Makefile rule for nl_forth_interpreter_vm

## [3.3.0] - 2026-03-23

### Added
- complete roadmap features 1-11 with self-hosting pipeline
- dispatch module + parallel physics in sdl_boids and sdl_falling_sand
- add automatic zero-boilerplate concurrency via libdispatch
- add intrinsic PEG grammar support (std/peg2)

### Fixed
- make test use nanoc_c; remove debugging artifact test
- pre-populate module-level float arrays with 0.0 to set ELEM_FLOAT type
- resolve peg2 crashes, union match bindings, and example build failures

## [3.2.0] - 2026-03-11

### Added
- Local type inference: `let x = 42` infers type from RHS (no annotation required)
- Pipe operator `|>`: `x |> f |> g` desugars to `(g (f x))` for readable chains
- String interpolation `f"..."`: `f"Hello {name}!"` desugars to str_concat at compile time
- Tuple destructuring: `let (q, r) = (divmod 17 5)` binds each element directly
- Wildcard `_` in match: catch-all arm `_ => { ... }` for exhaustive pattern matching
- Anonymous functions (lambdas): `fn(x: int) -> int { return (* x 2) }` as expressions
- `for x in List<T>`: iterate List<int>, List<string>, and List<struct> directly
- `--emit-typed-ast-json`: compiler flag emitting type-annotated AST as JSON for tooling
- All 8 features implemented in both C reference compiler and NanoLang self-hosted compiler

## [3.1.12] - 2026-03-07

### Fixed
- harden module dependency auto-install across platforms
- replace static buffers in fs.c path functions; fix gc_mark for ELEM_STRUCT arrays
- make examples now builds SDL/NCurses/network examples correctly
- correct array broadcast misidentification of float literals as identifiers

## [3.1.11] - 2026-03-04

### Added
- add module metadata support to stage1 compiler

### Fixed
- resolve make examples hang and black-on-black docs styling
- work around two stage1 transpiler bugs in examples and launcher
- restore original PNG icons for example launcher
- resolve 3 pre-existing stage1 transpiler failures
- eliminate 152+ -Wparentheses-equality warnings in self-hosted transpiler

## [3.1.10] - 2026-03-01

### Fixed
- resolve make examples build failures

## [3.1.9] - 2026-03-01

## [3.1.8] - 2026-03-01

## [3.1.6] - 2026-02-25

### Fixed
- eliminate const-qualifier warnings across build pipeline

## [3.1.5] - 2026-02-23

### Added
- add inline source editor with text input and syntax highlighting

## [3.1.4] - 2026-02-23

### Added
- add default args support for examples

## [3.1.3] - 2026-02-23

### Fixed
- run from repo root so icons and source code resolve

## [3.1.2] - 2026-02-23

### Changed
- gate verbose/debug output behind --verbose flag

## [3.1.1] - 2026-02-22

### Added
- infer anonymous struct literal names from function parameter types

### Changed
- rewrite SDL launcher with modular architecture
- replace str_concat with + operator in stdlib/timing.nano, update stale TODOs

### Fixed
- resolve conflicting types and missing nl_get_time_ms in stdlib
- eliminate all -Wdiscarded-qualifiers warnings from clean build and bootstrap
- resolve remaining TODOs in self-hosted compiler
- implement outstanding TODOs across compiler and examples
- reject pure expression statements, validate function arg types in self-hosted typechecker

## [3.1.0] - 2026-01-31

### Added
- shadow test audit: add ~167 missing shadow tests across compiler files
- generic union type support in test suite

### Fixed
- reject pure expression statements in self-hosted typechecker
- validate function argument types in self-hosted typechecker
- implement outstanding TODOs across compiler and examples
- resolve remaining TODOs in self-hosted compiler

### Changed
- replace str_concat with + operator in stdlib/timing.nano
- consolidate generic_union tests

## [3.0.2] - 2025-12

### Fixed
- bootstrap stability improvements

## [3.0.1] - 2025-12

### Fixed
- self-hosting bootstrap fixes

## [3.0.0] - 2025-12

### Added
- true 100% self-hosting achieved
- complete 3-stage bootstrap verified
- NanoLang compiler written entirely in NanoLang
