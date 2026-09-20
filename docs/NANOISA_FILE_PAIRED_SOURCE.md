# My paired File declarations, lowering and selected shadows

I propose this next source checkpoint under existing
`task_8bbc1cf5295b4b59b314640ef57c725f` and parents6fc/72556/6931/d03c. This is a
preimplementation contract, not source acceptance. I base it on publisher
review head742b1494d. My strict document/renderer is merged at
`af8809b32850454d06d9c1881c3a27b43f4c9d9c`; my independent descriptive C/Nano
plans are merged at `ec51690f7028fc788145d4d84e630627a07df668`. Publisher PR908
is still under review when I write this checkpoint. I preserve its frozen
source and evidence trees.

I reuse [my source publication requirements](NANOISA_FILE_SOURCE_PUBLICATION.md),
[strict binding vocabulary](NANOISA_FILE_BINDING_PUBLICATION_NEXT.md) and
[public byte/package conjunction](NANOISA_FILE_PUBLIC_ADMISSION_PLAN.md).
The generated source is currently forward text. Passing its renderer golden
comparison does not mean that either compiler can parse or execute it.

## My declaration and identity

I accept exactly the published declaration form in the first paired slice:

```nano
service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"
```

I recognize `service` contextually at top level and `catalog` only in this
production; ordinary identifiers with these spellings keep their old meaning
elsewhere. I require one string, the catalog identifier, the exact decimal
catalog version, `from`, and one nonempty relative document path, with no
trailing declaration tokens. I diagnose unsupported versions explicitly.
Parsing performs no I/O and does not grant authority. A declaration is not an
extern function, generic resource struct, ordinary union or magic filename.

The resolver opens the companion document relative to the declaring source
file's retained origin, never the flattened root file or current working
directory. It takes one bounded immutable snapshot and uses the qualified
strict File document preparation. It rejects missing, symlinked, nonregular,
oversized, malformed or mismatched documents before any service, loader or
output publication. I retain the input-path and stable-ancestor preconditions
of the reviewed publisher rather than claiming protection from hostile parent
replacement. I retain the original byte count through decoding, complete
catalog equality and preparation; I never reopen the pathname at lowering or
execution. A changed file after snapshot cannot change this compilation.

I build the existing thirteen bindings through `NlFileSourceRequest` /
`FileSourceRequest`: eight exact nominal types and five methods. Module origin,
interface ID, catalog version/view, binding kind/ordinal, source span and unique
global declaration ID are all significant. I apply the qualified alias,
namespace and ordinary-declaration collision checks. An alias resolves to the
same original identity; a different module with the same visible names does
not inherit authority. Import merging must retain original module identities,
including transitive imports and reordered declarations. Two snapshots claiming
one identity must agree completely or I refuse the compilation.

My C catalog bridge may expose immutable counted data and snapshot ownership;
it cannot return lowered instructions, resolve Nano AST nodes, choose layouts
or delegate Nano checking to C. Both producers independently resolve requests
and assign their own deterministic wire maps. Comparison fixtures normalize
source IDs only through an explicit retained mapping, never by dropping them.

## My first parser/schema dependency

My current generated `LexerToken` has token type, value, line and column but no
decoded byte count. `strlen` cannot distinguish a truncated embedded NUL from a
valid service string. Before service parsing I extend the primary compiler
schema with a decoded-byte-count field (`value_bytes`) and propagate it through
both lexers, token constructors, list/FFI bridges and generated C/Nano types.
I keep the complete decoded count even when the current Nano string runtime
cannot represent all those bytes. Service parsing refuses an inconsistent
extent or embedded NUL; it does not silently shorten the declaration. This is
a source metadata prerequisite, not a claim of new general string semantics.

I add a dedicated service declaration node/table with counted interface and
path text, version, source location and retained origin index. I extend C AST
free/copy/clone/export paths and every Nano Parser constructor, setter, import
merge and generated accessor. The parse node owns its text; a later resolved
binding owns its immutable snapshot and copied identity facts. It does not
retain a mutable token, borrowed document tree or host grant. Failed parsing
and resolution release every C allocation prefix without publishing the AST or
resolved output. Nano process allocation failures remain a separate runtime
limitation; I do not claim paired recoverable OOM. The outer producer owns and
removes all staged files after an unsuccessful process.

I inventory schema generators and generated consumers before assigning any
numeric parse-node/type values. I integrate the reviewed shared ownership
schema first: peer893's acknowledged8fadd11fd design uses v3 `path_bytes` as an
exact v2 substream followed by unique ordered TLVs (kind1 UNION_VARIANTS rev1,
kind2 ARRAY_FIELDS rev1). Acknowledgement is established; canonical integration
and producer/consumer review remain prerequisites. I neither reserve another
v3 grammar nor treat ordinary-union/array facts as File nominal authority.
Existing v1/v2 rules and unknown-feature/kind refusal remain intact. File's
required exact catalog/ownership metadata remains distinct. A coexistence
fixture must retain both sets of facts or explicitly refuse an unsupported
combination before execution; it must not silently drop either set.

## My checking and lowering boundary

I introduce resolved service callables and nominal types, rather than injecting
five fabricated ordinary extern functions. A service shadow binds to the exact
resolved callable ID. The spelling `temp` by itself grants nothing. I require
matching types across assignments, arguments, returns, aliases and imports;
File and affine OpenResult cannot be forged by an ordinary constructor or
copied as scalar storage. The seven-field FileError and ReadByte records retain
exact nominal identities and field types. Field reads, scalar Result matching
and ownership checks must agree in C and Nano.

I parse and check the renderer's real `Ok()` unit arm, `Ok(file)` affine arm,
`Error(error)` scalar arm and `&mut owned` borrow syntax. Unit is not a fake INT
payload. OpenResult refinement transfers the one owner; its Error branch owns
none. The exact operation contract remains:

| Operation | File transition | Result |
| --- | --- | --- |
| temp | New File exists only in the successful affine Result | OpenResult |
| write_byte | Borrow; checked ordinary INT domain0..255 before host access | Scalar WriteResult |
| rewind | Borrow; origin retained | Scalar PositionResult |
| read_byte | Borrow; origin retained | Scalar ReadResult, byte0..255 or canonical0 at EOF |
| close | Consume on success and reported close error | Scalar CloseResult |

I check all function bodies, including uncalled helpers, and both reachable
Result outcomes. I preserve pending cleanup obligations, reference regions,
nonduplication, same-call reference restrictions and exit obligations. I do not
accept a program solely because its descriptive request plan is valid.
Unsupported ownership/container combinations refuse before output; later
required language coverage remains on the parent roadmap.

C and Nano lower independently to the same required File v2 transport contract:
exact service120-byte bindings, FFI/import-presence bit1 plus service bit9,
nominal global/source IDs, import identities, layout flags, function signatures,
locals, reference slots, instruction targets and truthful wire max_stack.
Bit1 never grants generic FFI dispatch. I use the existing verified decoder,
nominal/flow/body and hosted conjunction as consumer validation, not as a C
source-lowering shortcut. Failed validation publishes no bytecode or C file.
Each producer preserves full serialized facts on roundtrip and deterministic
startup order. I do not emit NASM text and hope that an ordinary assembler can
supply missing service authority.

My first executable source route is the qualified acyclic direct-call profile
with its existing single-reference call boundary. Loops/backedges, indirect
calls and richer borrowed-call transport remain mandatory5.1 dependencies.
The separately owned cyclic public design will use a distinct API and explicit
fuel contract; this source checkpoint does not infer a fuel policy, reinterpret
the acyclic ABI or select private cyclic plans. Later source selection must be
reviewed against actually qualified matched targets for every accepted case.

## My actual producer integration points

| Producer | Current path I must change or guard |
| --- | --- |
| C seed | `src/parser.c::parse_program/parse_shadow`, handwritten AST and checker; `src/main.c` before `compile_modules`, interpreted shadow execution and C transpilation |
| C byte/native CLI | `src/nanovirt/main.c` before `build_ffi_modules`, ordinary `check_shadows`, ordinary codegen and FFI import binding |
| Nano Stage1/2 | `src_nano/parser.nano::parse_definition/parse_program/parse_shadow`, generated Parser tables and module-origin merging; `src_nano/nanoc_v06.nano::compile_program` before ordinary NASM shadow emission/`--check-shadows` and C transpilation |

My C typed-AST export/reflection and any other dropping consumer must retain the
new complete facts or refuse a service graph explicitly. I audit reusable entry
APIs, not only CLI branches. Nano merged-source origins must survive
`merge_with_imports`; a root line threshold alone is not a module identity.
The data bridge remains usable by both bootstrap stages without dispatching
service operations while loading the compiler's modules.

I add explicit source-driver `--allow-temporary-files` opt-in and preserve its
meaning through the supervisor and selected backend. With no opt-in, service
execution, shadow execution and executable publication refuse before host
acquisition or generic loader attempts. Nonexecuting parse/inspection tests use
explicit internal APIs and do not imply that executable publication may skip
mandatory shadows. Legacy C transpilation, interpreter, COP/daemon wrappers,
generic FFI and LLVM/Wasm routes refuse a service graph before effects until
separately supported and qualified. I test actual attempt counters, including
failed opens/loading, rather than only an unchanged initialized flag.

For VM I pass the final immutable bytes and explicit grant to
`nvm_file_execute_bytes`. For native I emit through `nvm2c_emit_file_bytes` with
a validated collision-free entry identifier and call the generated public
entry with the same explicit grant contract. Native output contains compiled
functions/labels, not a private VM interpreter. Multiple generated programs
share one owning C11 grant/gate object and cannot create separate concurrent
authority through header copies. I retain the installed archive/header closure
and ordinary C99 consumer ABI.

## My selected shadows, startup and publication

I keep all five published behavioral shadows unchanged in meaning: temp and
close lifecycle; write0/255 with exact progress; rewind then read255; read0 then
EOF with canonical0. I add negative byte bounds, consumed-close-error and
failed-init cleanup controls in qualification rather than weaken these
positive shadows. An Error arm's assertion failure still drains every live
File/root before the supervisor reports failure.

Every selected service shadow gets a fresh invocation/context and explicit
host grant; I check grant destruction and complete cleanup status. I preserve
exact VOID/no-result initializer signatures and ordered initializer execution.
Initializer failure or cleanup failure prevents entry and later shadows. Public
entries escape only clean INT/BOOL scalars; no File, affine Result, reference or
context escapes. I preserve full64-bit API results and existing CLI low-byte
exit conventions. I report first and secondary cleanup failures separately.

I select dependency and root shadows by default, including the five generated
service shadows and every new compiler/helper/module shadow. I retain an exact
normalized selected-name-and-origin multiset for C seed, fresh Stage1 and fresh
Stage2, alongside actual starts/completions/outcomes. Counts or a few substring
matches are insufficient. The explicit existing root-only option is tested as
a distinct reduced selection, never used to claim full acceptance. A service
binding lacking its required behavioral shadows cannot take an ordinary extern
exemption. Shadow/initializer startup identity must match the module actually
executed; no shadow-only unchecked reconstruction or mutable-module fallback.

I preserve the existing whole-suite supervisor deadline and durable bounded
process-group logging. A process timeout, signal, allocation failure, failed
shadow or report/cleanup failure prevents final output publication. I stage
bytecode/native outputs under producer ownership, forbid collisions with every
source/companion path, and publish only after all selected shadows and required
validation succeed. I retain existing destination sentinels on failure. My
exclusive binding-directory publisher's earlier commit is separate: compiling
its files does not retroactively remove or relabel that committed directory.

## My review and acceptance order

1. I review complete token-count/schema/AST ownership and import-origin changes,
   including primary/generated consistency, unknown declarations and legacy
   adjacency. I qualify paired nonexecuting parser tests before source effects.
2. I review exact resolver/type/alias/Result/shadow binding and allocation bounds.
   I test counted NUL/non-ASCII/partial UTF-8, malformed and mutated companions,
   duplicate declarations, same names in different modules, input lifetime,
   every C allocation prefix and Nano process failure/staging containment.
3. I review both independent lowerers and compare complete retained modules from
   C seed/Stage1/Stage2: all catalog/signature/layout/ownership/import/startup
   facts, instruction behavior, malformed/refused cases and ordinary neighbors.
   No host execution precedes review of complete production and fixtures.
4. I qualify actual grant-driven VM/native execution of the published source and
   every selected shadow on Linux/Darwin. I cover both Result arms, host faults,
   borrow/move/close/initializer failures, no-host missing-grant refusal and
   clean-only output. Sanitizer evidence names instrumented provider scope.
5. I run fresh C-seed→Stage1→Stage2 bootstrap with all new helper shadows and the
   complete generated module, then real installed outside-tree compilation,
   execution and companion resolution without repository-header/library fallback.
   I preserve canonical module/output comparisons, source/tool/provider hashes,
   all first terminals and every generated/native artifact at the actual pin.
6. I integrate the separately qualified cyclic/indirect/richer-borrow source
   routes and rerun affected full source/shadow/product gates. These and the
   original File/Socket/GPU parent requirements are not waived by steps1–5.

This document changes no parser, schema, runtime, public selector or generator.
Source implementation requires review of this contract and the exact first
bounded checkpoint. I do not close8bbc or any full File/source parent here.
