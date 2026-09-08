# 5.0 — I emit one thing

This is the compilation contract for release **5.0**. It is a major-version
rewrite of how I become a process image. It is not 4.x work. 4.0 keeps the
decision and a closed-subset spike (`nvm2c` for i64 arithmetic). 4.x keeps
Forth, internationalization, services, capabilities, effects, and the
language laboratory. I do not delete `transpiler.nano` until this contract
has compiled me.

I speak plainly: C is a substrate, not my IR. NanoISA is my IR. Every
frontend lowers to one verified module. Every native or portable target
is a **consumer of that module**. The NanoLang→C pretty-printer leaves
the compiler. `cc` stays. `nano_vm` stays as a runner, not as the
definition of "native."

## The shape

```
source (NanoLang, Forth, later others)
        |
        |  frontend: lex, parse, typecheck, modules
        |  (exists twice: C seed and src_nano)
        v
   verified .nvm v2
        |
        +---> nano_vm          interpret / debug / Forth session
        +---> nvm2c → cc       canonical AOT portable native
        +---> nvm2llvm         optional translator
        +---> nvm2wasm         optional translator
        +---> nvm2riscv        optional translator
        +---> GPU translators  restricted compute profile only
```

There is no arrow from my AST to C, LLVM, or Wasm in the product compiler.
Those arrows were the matrix I am done paying for.

`nano_virt -o` today still embeds `.nvm` plus `nano_vm` (`wrapper_gen.c`),
and may talk to `nano_cop` or `nano_vmd`. That is a **packaged interpreter**.
It is allowed as a debug and Forth-session path. It is not a native
binary in the 5.0 sense. A 5.0 native process computes with C operators
(or LLVM machine code, or Wasm) and does not require `nano_vm`,
`nano_cop`, or `nano_vmd` to add two integers.

## Bootstrap, rewritten

I already compile myself from C: `cc` builds `bin/nanoc_c` from `src/`.
Self-hosting then pretty-prints C twice. Stage 3 compares native binaries
and apologizes for UUID noise. That is a translator property pretending
to be a compiler property.

5.0 bootstrap:

```
Stage 0  cc builds nanoc_c (frontend + NanoISA codegen) and nvm2c
Stage 1  nanoc_c --emit-nvm src_nano → stage1.nvm
         nvm2c stage1.nvm → stage1.c → cc → nanoc_stage1
Stage 2  nanoc_stage1 --emit-nvm src_nano → stage2.nvm
         nvm2c stage2.nvm → stage2.c → cc → nanoc_stage2
Stage 3  compare stage1.nvm and stage2.nvm
```

Stage 3 becomes honest. `.nvm` is what I generated. If those match, I
compiled myself. `nvm2c`+`cc` matching is a **translator** test, kept
separate: same module through the VM and through AOT C, same observable
results on a pinned suite.

The seed `nvm2c` stays C, the way `cc` stays C. I may later write
`nvm2c` in myself and lower it through NanoISA. The seed translator
remains a host program. Runtime C (`gc.c`, lists) is the same pattern:
not a language feature, not 2×.

## What `src_nano` becomes

**Keep.** Lexer, parser, typechecker, diagnostics, module loader, `file_io`,
CLI, generated AST/schema. Those do not care what the last pass emits.

**Replace, not dual.** `transpiler.nano` (~7.6k lines of C emission) ceases
to be a compiler phase. Its dual with `src/transpiler.c` is the tax. The
last pass becomes a dual of `nanovirt/codegen.c`: typed AST → `NvmModule`
→ `.nvm`. That dual does not exist in `src_nano` today. That is the real
self-hosting work of 5.0.

**Driver.** `nanoc_v06.nano` stops emitting `.c` and invoking `cc` as a
language backend. Default output is `.nvm`. `-o binary` is the tool
pipeline `emit .nvm → nvm2c → cc`.

**Schema.** `CompilerPhase_PHASE_TRANSPILER` is the wrong name. The
pipeline ends at NanoISA. Target translators are not compiler phases.

**IR.** `compiler/ir.nano` grows NanoISA module types, or those live next
to codegen. It does not grow a C AST.

**What I will not put in `src_nano`.** `nvm2c`, `nvm2llvm`, `nvm2wasm`,
`nvm2riscv`. Those are ISA tools.

## Dual implementation after the move

Today a language feature is 2× because **frontend and C emission** exist
twice.

After 5.0 a language feature is 2× because **frontend and NanoISA
lowering** exist twice. The cost moves. I stop duplicating `nl_` mangling,
`gc_release` placement, and include graphs. I start duplicating bytecode
lowering (locals, jumps, imports, layouts) which so far exists only in C.

Adding a **target** is not 2× in `src_nano`. It is one ISA consumer. That
is the reason LLVM frontends do not each emit C.

## The module as a product

v2 already has `FUNCTIONS`, `SIGNATURES`, `LAYOUTS`, `IMPORTS`, `LINKS`,
`CONSTANTS`, `METADATA`, `DEBUG`. 5.0 makes that the *only* portable
compiler product:

- **Local names**, not only slot numbers, so reconstructed C and NanoLang
  are readable. Bootstrap does not need pretty names. A C backend as a
  product does.
- **Frontend facts** as metadata: purity, affine-use, generics, effects,
  exhaustiveness. Verifier-enforced where they affect safety; optional
  where they only affect reconstruction or optimization.
- **Declared host ABI** for imports. `CALL_EXTERN` as RPC into `nano_cop`
  is a VM isolation story. AOT C emits a call into a fixed C ABI, or
  it refuses the module. I do not pretend those are the same.
- **Compute profiles.** General (VM, AOT CPU) and restricted (GPU
  kernels). The verifier enforces the feature set. I do not map every
  general opcode onto a kernel.
- **Structured control** recovered from `JMP` / `JMP_FALSE` into
  `if`/`while`/`return` for reconstruction. Goto is a translator
  fallback, not the reconstruction claim.

Reconstruction succeeded when a reader of one `.nvm`, without the
original source, can produce named functions, types, structured control,
and a host ABI. A C file that embeds `nano_vm` plus a bytecode array
failed. Canonical disassembly is a different test.

## Translators

| Translator | Job | Not |
| --- | --- | --- |
| `nvm2c` | Canonical AOT portability. C11. Locals, arithmetic, calls, control, runtime ABI. | VM wrapper |
| `nvm2llvm` | Optional. Same module, LLVM IR. | NanoLang AST backend |
| `nvm2wasm` | Optional. Same module, Wasm. | NanoLang AST backend |
| `nvm2riscv` | Optional. Same module, assembly. | AST `riscv_backend` as a second IR |
| GPU (SPIR-V / PTX / Metal / OpenCL) | Restricted profile only | Pretending general NanoISA is a kernel language |
| JVM / others | Evaluate; accept only as translators of verified NanoISA | New AST backends |

I reintroduce LLVM and Wasm only behind these translators, with the same
module run through the VM and through the translator for semantic
equivalence.

PTX and OpenCL AST backends that still hang off `nanoc --target` either
become NanoISA→target translators under the restricted profile, or they
leave the product compiler. Mixing the two stories recreates the matrix.

## Runtime and isolation

`nano_vm` remains the reference interpreter and the Forth session host.
`nano_cop` remains process-isolated FFI for the VM path. `nano_vmd` remains
an optional daemon for that path.

5.0 AOT binaries do not spawn those processes to compute. Host calls go
through the declared ABI. If a module needs isolation, 4.4's capability
fabric supervises it; isolation is not a substitute for a C backend.

The tree-walker (`bin/nano`) is a third semantics. Bootstrap does not
need it. Keeping it is a product choice.

## Forth already lives here

Forth on NanoISA already compiles words to verified functions and runs
them in one session. 5.0 makes NanoLang the same citizen: one module
format, one verifier, one set of translators. Forth does not grow a C
pretty-printer.

## Why this is the architecture

I am not a C generator with a VM sidecar. I am a compiler family with
one verified instruction set. LLVM proved that a mid-level IR can host
many languages and many machines. I take the same shape, with a
narrower contract:

- the IR is a **module**, not a pile of SSA files
- the verifier is **mine**, not an after-the-fact optimizer
- frontends exist **twice** (C seed and `src_nano`) until I trust the
  lowering, then the last pass is NanoISA on both
- targets are **host tools**. They do not live in the language compiler
- "native" means **operators in the process**, not a bytecode blob
  wearing a process costume

The dual-implementation tax does not disappear. It **moves**: every
language feature is still lexer/parser/types plus NanoISA lowering,
twice. What disappears is the third and fourth copy of every feature
as C pretty-printing, LLVM-from-AST, Wasm-from-AST, and GPU-from-AST.
That matrix is how I paid twice for the language and N times for
targets. 5.0 pays twice for the language and once per translator.

## How I walk there

I do not start 5.0 by deleting the C path. I walk it in named cuts.
Each cut has a test that can fail without stranding bootstrap.

**A — Emitter exists.** `src_nano` grows a dual of
`src/nanovirt/codegen.c`. It emits `.nvm`. The C pretty-printer still
builds the compiler. I compare `.nvm` from the C seed's NanoISA path
and from `src_nano` on a pinned subset, not yet the whole compiler.
Cut A pin: `src_nano/compiler/nanoisa_codegen.nano` plus
`make test-nanoisa-src-nano` on integer `add`/`main`/`choose`/`loop_sum`,
string `greeting`/`glue`, array `len3`/`first`, record `getx`, bool
`is_pos`, bool literals `yes`/`no`, `invert`, `both`, `either`,
`pick`, `say`, `shout`, `mutter`, `prove`, `grow`, `has_hi`,
`digits`, `names`, `head_s`, `same`, `diff`, `via_at`, `slen`,
`slice`, `blank_l`, `grow_l`, `ch`, `blank_s`, `grow_s`, `get_s`,
`blank_t`, `grow_t`, `get_v`, `grow_lex`, `has_pre`, `has_suf`,
`put_l`, `put_t`, `put_s`, `upto`, `quiet`, `via_quiet`, `origin`,
`via_o`, `make_tok`, `via_tok`, `ones`, `via_ones`, `new_l`,
`via_new_l`, `one_t`, `via_one_t`, `one_lex`, `via_one_lex`,
`in_az`, `via_az`, `tag`, and `via_tag`
(`make test-nanoisa-src-nano`, 134 passed).
Function bytecode matches the C seed, including `if`, `while`, `PUSH_STR`,
`STR_CONCAT`, `ARR_LITERAL`, `ARR_LEN`, `ARR_GET`, `ARR_PUSH` of
`array<int>` and `array<string>`, `AGG_PACK`, `AGG_GET`, `bool`
results as i64 0/1, `PUSH_BOOL`, `BOOL_NOT`, `BOOL_AND`, `BOOL_OR`,
`cond` as `JMP_FALSE`/`JMP` with one `RET`, `PRINT`, `PRINTLN`,
`ASSERT`, `STR_CONTAINS`, `CAST_STRING` of i64, `EQ`/`NE` of
strings, `at` as `ARR_GET`, `str_length` as `STR_LEN`,
`str_substring` as `STR_SUBSTR`, `list_int_new` as `ARR_NEW 1`,
void `list_int_push` as `ARR_PUSH` then `POP`, `list_int_get`
as `ARR_GET`, `char_at` as `STR_CHAR_AT`, `list_string_new` as
`ARR_NEW 1`, void `list_string_push` as `ARR_PUSH` then `POP`,
`list_string_get` as `ARR_GET`, and mixed int/string records as
`AGG_PACK`/`AGG_GET`, and `List<Tok>` as `ARR_NEW 1` / `ARR_PUSH`
then `POP` / `ARR_GET`, and `LexerToken` plus `List<LexerToken>` as
`get_v`/`grow_lex`, `str_starts_with` as `STR_STARTS_WITH`, and
`str_ends_with` as `STR_ENDS_WITH`, `list_int_set` as `ARR_SET`
then `POP`, `list_Tok_set` as `ARR_SET` then `POP`,
`list_string_set` as `ARR_SET` then `POP`, `for` over
`array<int>` as `LT`, and `void` as `RET` with no value
(`CALL quiet` does not `POP`), and pin-record results as ISA tag
`struct` (`origin` / `via_o`), and mixed int/string record results
(`make_tok` / `via_tok`), and array results as ISA tag `array`
(`ones` / `via_ones`), and `List<int>` results (`new_l` / `via_new_l`),
and `List<Tok>` results (`one_t` / `via_one_t`), and
`List<LexerToken>` results (`one_lex` / `via_one_lex`), and i64
`>=`/`<=` (`in_az` / `via_az`), `Enum.Variant` as `ENUM_VAL`
(`tag` / `via_tag`), and imported `Enum.Variant` (`tok_mod` /
`via_tok_mod`; local enums stay first in `def_idx`). `std/` import
paths resolve through `modules/` like the C seed. Transitive
imports register enums and pin-record structs from imported files.
`STR_SPLIT`
and `STR_REPLACE`
stay refused. String operands
are compared by content, not pool index. I still pretty-print C to
build the compiler. When I refuse a program, `nanoisa_emit` prints
`nisa_err` so the next hole is named. The full dual of `codegen.c` is
not this pin.

**B — AOT covers the compiler subset.** `nvm2c` translates functions,
structs, loops, arrays, strings, modules, and a declared host ABI.
`CALL_EXTERN` maps to that ABI or the module is refused. I ship a
`nvm2c` CLI (`bin/nvm2c`, `make test-nvm2c`). The closed i64 subset
already builds a process that does not link `nano_vm`, including
comparisons, `JMP`/`JMP_FALSE`, `TAIL_CALL` (`choose` and `loop_sum`
run as native C), Cut A strings (`PUSH_STR`, `STR_CONCAT`,
`STR_LEN`; `greeting` and `glue` run as native C), Cut A
`array<int>` (`ARR_LITERAL`, `ARR_GET`, `ARR_LEN`; `len3` and `first`
run as native C), Cut A int-field records (`AGG_PACK`, `AGG_GET`;
`getx` runs as native C), Cut A bool (`bool` results as i64 0/1;
`is_pos` runs as native C), Cut A bool ops (`PUSH_BOOL`,
`BOOL_NOT`, `BOOL_AND`, `BOOL_OR`; `yes`, `invert`, `both`, and
`either` run as native C), Cut A `cond` (`pick` runs as native
C; join copies temps so both arms share one `RET`), Cut A
print (`PRINT`/`PRINTLN`; `say`, `shout`, and `mutter` run as
native C), Cut A assert (`ASSERT`; `prove(true)` exits 0 and
`prove(false)` aborts), Cut A `array_push` (`ARR_PUSH` of
`array<int>`; `grow` runs as native C), Cut A `str_contains`
(`has_hi` runs as native C), Cut A `int_to_string`
(`CAST_STRING` of i64; `digits` runs as native C), and Cut A
string arrays (`ARR_LITERAL` tag 5, `ARR_PUSH`, `ARR_GET`, `ARR_LEN`;
`names` and `head_s` run as native C), Cut A string equality
(`EQ`/`NE`; `same` and `diff` run as native C), and Cut A
`at`/`str_length` (`via_at` and `slen` run as native C), and Cut A
`str_substring` (`STR_SUBSTR`; `slice` runs as native C), and Cut A
empty `List<int>` (`ARR_NEW`; `blank_l` runs as native C), and Cut A
void `list_int_push` (`ARR_PUSH` then `POP` keeps array identity;
`grow_l` runs as native C), and Cut A `char_at` (`STR_CHAR_AT`;
`ch` runs as native C; out of range is `-1`), and Cut A empty
`List<string>` (`ARR_NEW` tag 1; `blank_s` runs as native C), and
Cut A void `list_string_push` (`ARR_PUSH` then `POP` keeps string-array
identity; `grow_s` runs as native C), and Cut A mixed int/string
records (`AGG_PACK`/`AGG_GET`; `get_s` runs as native C; nested
records stay refused), and Cut A lists of records (`ARR_PUSH` of
`nrec_t`; `blank_t` and `grow_t` run as native C; I classify
`ARR_NEW 1` from the pushed record), and Cut A `LexerToken` lists
(`get_v` and `grow_lex` run as native C), and Cut A prefix and
suffix tests (`STR_STARTS_WITH`/`STR_ENDS_WITH`; `has_pre` and
`has_suf` run as native C), and Cut A `list_int_set` (`ARR_SET` of
`array<int>`; `put_l` runs as native C), and Cut A `list_Tok_set`
(`ARR_SET` of a record list; `put_t` runs as native C), and Cut A
`list_string_set` (`ARR_SET` of a string list; `put_s` runs as
native C), and Cut A `for` (`LT`; `upto` runs as native C), and Cut A
`void` (`quiet` / `via_quiet` run as native C), and Cut A pin-record
results (`origin` / `via_o` run as native C; nested records stay
refused), and Cut A mixed int/string record results (`make_tok` /
`via_tok` run as native C), and Cut A array results (`ones` /
`via_ones` run as native C), and Cut A `List<int>` results (`new_l` /
`via_new_l` run as native C), and Cut A `List<Tok>` results (`one_t` /
`via_one_t` run as native C; nested records stay refused), and Cut A
`List<LexerToken>` results (`one_lex` / `via_one_lex` run as native C),
and Cut A i64 `>=`/`<=` (`in_az` / `via_az` run as native C), and Cut A
`ENUM_VAL` (`tag` / `via_tag` run as native C; nested unions stay
refused), and Cut A imported `ENUM_VAL` (`tok_mod` / `via_tok_mod` run as
native C; transitive imports and import-before-local-enum order stay
refused), and Cut A one-level nested pin-record field types
(`List<CompilerDiagnostic>` parameters type-check; nested `AGG_PACK`
stays refused), and Cut A imported functions (`imp_add` / `via_imp_add`
run as native C; `CALL`/`TAIL_CALL` compare by callee name), and Cut A
`unsafe` blocks (`via_raw` run as native C; inner statements only), and Cut A host `CALL_EXTERN`
(`via_cwd` matches the C seed by import symbol; `nvm2c` still refuses
`CALL_EXTERN`), and Cut A synthetic `main` for library files
(`PUSH_I64 0` then `RET`, the way the C seed does), and Cut A `.string`
payload that contains `;` or `#` (`lexer.nano` nasm assembles), and Cut A
nested pin-record literals (`nest_d` / `via_nest` match the C seed;
`nvm2c` still refuses nested records), and Cut A bool and `List<int>`
record fields (`flag_yes` / `via_flag`, `empty_bag` / `via_bag`), and Cut A
string `+` of a field or `int_to_string` (`glue_field` / `glue_digits`) and
`str_concat` as `STR_CONCAT` (`glue_sc`), and Cut A `string_to_int` as
`CAST_INT` (`parse_n` / `via_parse_n` match the C seed; `nvm2c` runs
`via_parse_n` without `nano_vm`), and Cut A nested field access as chained
`AGG_GET` (`nest_line` / `via_nest_line` match the C seed), and Cut A
module-qualified call (`via_q_add` / `(ImpMod.imp_add 40 2)` match the C
seed; `parser.nano` emits 297 functions and that nasm assembles), and Cut A
`array<Loc>` (`empty_locs` / `via_empty_locs` / `one_loc` / `via_one_loc`
match the C seed; empty `[]` is `ARR_LITERAL 1 0`), and Cut A HashMap
(`blank_hm` / `via_blank_hm` / `put_hm` / `via_put_hm` match the C seed;
`map_new` is `HM_NEW 5 1`), and Cut A `array<Sym>` params (`n_syms` /
`via_n_syms` match the C seed; a pin record may have `array<pin-record>`
fields), and Cut A interned string escapes (`quoted` / `via_quoted` match
the C seed; `"a\"b"` is `.string` with `\"`), and Cut A bool `==` as `EQ`
(`flag_is_ok` / `via_flag_is_ok` match the C seed). `typecheck.nano`
emits 401 functions and that nasm assembles; `nano_vm` exits 0. `nvm2c`
still names `CALL_EXTERN`. Cut A `array_new` (`blank_a` / `via_blank_a`
match the C seed; `ARR_NEW 1` even for a string fill), and Cut A
`str_substring` concat (`via_substr_concat`) and `EQ` of `at` of
`array<string>` (`via_at_eq`), and Cut A host `getenv` (`via_env` is
`CALL_EXTERN` `vm_getenv`), and Cut A module-level lets (`via_g_len` /
`via_g_set` / `__init__` match the C seed; `LOAD_GLOBAL` /
`STORE_GLOBAL`). Pin records may have `array<bool>` fields
(`empty_fb` / `flag_n`). Assembler `MAX_SYMBOLS` is 8192.
`module_loader.nano` emits 559 functions and that nasm assembles;
`transpiler.nano` emits 444. `nano_vm` exits 0. `nvm2c` still names
`CALL_EXTERN` and `LOAD_GLOBAL`. Host `tmp_dir` (`via_tmp`) is
`CALL_EXTERN` `vm_tmp_dir`. `file_read` / `file_write` / `file_exists`
lower as `vm_file_*`. `file_io.nano` emits 3 functions and that nasm
assembles. Bare void `return` (`via_bare`) is `RET` with no value.
`nanoc_integrated.nano` names `unsupported result type Vector2D`.
`compiler_modular.nano` names `system`. `nanoisa_codegen.nano` as a
root names `parse failed`. Labels reset per `.function`.
Nested arrays,
`AGG_SET`, variants, tuples, array
equality, `STR_TRIM`, `STR_SPLIT`, substring of arrays, and printing
arrays/records stay refused. The rest of the
compiler subset (modules, host ABI, and the remaining string
library) is still open. A pinned suite must match on `nano_vm` and
on AOT C.

**C — Product output is the module.** Self-hosted `nanoc --emit-nvm`
is the compiler. `-o binary` is `nvm2c | cc`, a tool pipeline written
down in the driver, not a language phase. `CompilerPhase_PHASE_TRANSPILER`
is renamed. `wrapper_gen` remains a packaged-interpreter flag, not the
default meaning of native.

**D — Honest bootstrap.** Stage 1 and Stage 2 `.nvm` files match. I
freeze `transpiler.nano` as bootstrap-only. Matching native binaries
are a translator test, kept separate.

**E — Pretty-printer leaves.** I delete `transpiler.nano` from the
product compiler (git keeps it). Optional `nvm2llvm` / `nvm2wasm` /
`nvm2riscv` consume the same module. GPU translators use the restricted
profile or they leave. I spike a second high-level surface from one
`.nvm` and publish sufficient / insufficient / blocked in
`docs/NANOISA_HL_ROUNDTRIP.md`.

I do not skip to E. A cut that cannot compile `src_nano` is not done.

## File fate

These are product decisions, not a cleanup pass:

| Today | 5.0 |
| --- | --- |
| `src/transpiler.c`, `src_nano/compiler/transpiler.nano` | Freeze, then remove from the product compiler |
| `src/c_backend.c` | Driver of `nvm2c`, or gone; not a second IR |
| `src/nanovirt/codegen.c` | Stays; `src_nano` grows its dual |
| `src/nanovirt/wrapper_gen.c` | Packaged interpreter / Forth-session native; not default `-o` |
| `src/nanoisa/nvm2c.c` | Canonical AOT; grows from the i64 spike to the compiler subset |
| `src/ptx_backend.c`, `src/opencl_backend.c` | NanoISA→GPU translators under the restricted profile, or they leave |
| `src/riscv_backend.c` | Becomes `nvm2riscv`, or it leaves |
| Retired AST LLVM/Wasm | Stay retired. Re-enter only as translators of `.nvm` |
| `nano_vm`, `nano_cop`, `nano_vmd` | Stay as VM path. AOT does not spawn them to add integers |
| `bin/nano` tree-walker | Product choice; not bootstrap |

## Linking, debug, and equivalence

v2 already refuses to flatten dependency modules into the root file.
5.0 AOT respects that: translators consume a linked module graph, not a
single mashed C translation unit, unless a translator documents a
flatten as its own lowering. `CALL_MODULE` stays a module index, the
same one `vm_link_named_module` already checks.

Debug information lives in the NanoISA `DEBUG` section and becomes
DWARF (or the target's analog) in the translator. I do not keep a
second debug pipeline on the AST.

Equivalence is a harness, not a slogan: the same `.nvm` on `nano_vm`,
on AOT C, and on each shipped translator, against a pinned suite.
Disagreement is a translator bug or a verifier hole. I do not paper
over it with "backend differences."

Reconstruction is the richness test for the module format. If a reader
of one `.nvm` cannot recover named functions, types, structured
control, and a host ABI, the IR is still a VM encoding. I will say so
in `docs/NANOISA_HL_ROUNDTRIP.md` instead of hoping.

## Completeness gate before I delete the pretty-printer

The self-hosted NanoISA emitter must compile **the compiler**: functions,
structs, loops, arrays, strings, modules, `extern` mapped to the host
ABI. Until that subset is closed, `transpiler.nano` stays as
bootstrap-only. I freeze it, then delete it once `stage1.nvm` is the
artifact that builds the next compiler.

I do not start 5.0 by deleting the C path. I start by writing the
NanoISA emitter in `src_nano` until it compiles `src_nano`.

## What 5.0 is not

- Not 4.2 catalogs, 4.3 service schemas, 4.4 capability fabric, 4.5
  replay, or 4.6 extra frontends. Those remain 4.x.
- Not 6.0 (the operating environment that was formerly numbered 5.0):
  signed images, kernel adapters, seL4. Those sit on top of this IR,
  they are not a substitute for it.
- Not a claim that I can reconstruct almost any high-level language
  until local names, structured control, and a host ABI exist and a
  second surface from the same `.nvm` is not an interpreter.

## Acceptance

5.0 closes when all of these are true:

1. `src_nano` emits `.nvm` as its only compiler product.
2. `nvm2c` translates that module to structured C11 that `cc` builds into
   a process which does not link `nano_vm`.
3. Stage 1 and Stage 2 `.nvm` files match.
4. A pinned suite gives the same answers on `nano_vm` and on AOT C.
5. `transpiler.nano` is gone from the product compiler (history keeps it).
6. LLVM and Wasm, if present, are translators of the same module.
7. `docs/NANOISA_HL_ROUNDTRIP.md` states sufficient / insufficient /
   blocked from evidence, not aspiration.

Until then I keep paying the C pretty-printer tax, on purpose.
