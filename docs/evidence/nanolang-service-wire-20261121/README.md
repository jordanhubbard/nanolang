# My service wire and byte-storage evidence

I implement byte-storage prerequisites at `f75aa6189` and my pure NanoLang
service codecs at `ff6aa99bd`, under #1142 and draft PR #1144. I change no
executable selection. My installed translator still comes from C; this is
progress toward its replacement, not a completed release.

## My raw codec contract

`src_nano/nvm2c/service_wire.nano` decodes complete bindings, checks normal
narrow carriers and emits canonical bytes without calling the C codecs.

| Profile | Version/catalog | Methods/types | Wire bytes |
| --- | --- | --- | --- |
| Legacy File | 1/1 | 5/0 | 56 |
| Nominal File | 2/1 | 5/8 | 120 |
| Nominal TCP | 2/2 | 5/9 | 128 |
| Nominal WebSocket | 2/3 | 4/7 | 104 |
| Multiple instances | 3/0 | Catalog-dependent active slots | 16 +64 per instance; maximum4112 |

The caller selects the exact profile. I preserve size/version/catalog/count/
reserved/ordinal/index precedence. Legacy offset8 is reserved, rather than a
nominal type count. I check all fixed-table ordinals before duplicate/sentinel
indices. V3 checks all entry framing first, then semantic facts in instance
order: catalog, unused layout/import slots, active imports and active layouts.
Active indices must be unique across instances within their own domain;
imports and layouts may reuse the same number. Unused slots retain UINT32_MAX.
I preserve1..64 instances, exact entry width and every original integer bound.

Failure returns no partial decoded values or encoded bytes. C pointer/null,
overlap, size-only and mutable destination semantics remain C API controls;
my immutable value model does not pretend to provide arbitrary pointer graphs.
Raw success grants no module, flow, execution or host authority. Section14
remains pending until its complete metadata/profile checks are implemented.

## My paired qualification

I match9208 complete decode/value/check/canonical-byte reports, with7311 explicit
status expectations, across five profiles and five routes on Linux ARM64 and
Darwin ARM64. The routes are C-seed native plus C/NanoLang producers through
NanoVM and ASan/UBSan-instrumented generated C. I also instrument all five
actual C codec providers, not only their wrappers. The oracle checks decode
failure atomicity, encode failure/short-buffer preservation and checked
size-only mode. Original C raw/nominal units retain overlap/null/allocation
controls beyond my pure value model.

The corpus retains every truncation of each base packet, all256 header-byte
values at five positions, byte/bit corruption, cross-profile packets, ordinal
and sentinel errors, unsigned extrema, independent uniqueness domains, inactive
slots, exact63/64/65-instance controls, size caps and refusal precedence across
rows. Value carriers exercise check/encode failures independently of decoding.

I retain fourteen actual modules from both producers: four computed/direct
byte-write variants in both function orders, plus the unchanged generic
scalar/array, original u8 and original byte-conversion fixtures. Every product
passes verification, real NanoVM execution and instrumented native execution.
Sources, module bytes, VM/native results and generated-C/native hashes are
retained. Four byte-write families print `255\n255\n`; the original fixtures
print nothing. Dependency shadows remain enabled, including the original file
write/read/delete operations in the byte-conversion fixture. Temporary native
images/generated C are not archived; their identities and results are retained.

| Gate | Linux ARM64 | Darwin ARM64 |
| --- | --- | --- |
| Raw codecs and instrumented providers | Two methods pass | Two methods pass |
| Byte-write/source/refusal/original fixtures | Four methods pass; five routes | Four methods pass; five routes |
| Original byte/nested/record-array controls | Eleven methods pass | Eleven methods pass |
| Original conversion/alias/argument controls | Eighteen methods pass with actual C producer | Eighteen methods pass with actual C producer |
| Original structured-C translator | 2458 pass, zero fail | 2458 pass, zero fail |
| Original callable/shape constraints | 505/3890 pass | 505/3890 pass |
| Original VM builtins / VM-FFI | 23/30 pass | 23/30 pass |
| Original byte literals, checker, inference, evaluator, metadata | Pass | Pass |
| Original five C raw/nominal targets | Pass, including allocation controls | Pass, including allocation controls |
| Capture/replay/self-capture/refusal controls | Twelve methods pass | GNU-as controls are Linux-specific |
| Embedded identifiers | Five methods pass | Five methods pass |

My byte-write source checks retain all36 prior-output refusals across three
producers, plus a malformed integer write into boolean-array native storage.
The original translator target retains its own platform skips and runtime
refusal probes; expected invariant messages in its log are not hidden failures.
Independent synthetic bytes/reports and449 source hashes match across hosts.
Fourteen actual product source hashes and VM/native outcomes also match.
Identifier checks retain3296 fragments, zero reserved names and135 existing
lexer-refused fragments whose executable tests retain responsibility.

## My retained failures and repairs

The first actual C-provider gate passes, but native translation of my unchanged
service fixture refuses conflicting byte/integer array storage. A smaller
computed-write witness reproduces this; replacing only `(+ number 0)` with the
parameter translates. Under #1150 I preserve unknown receiver storage until
caller facts converge. Exact final storage/write checks remain intact. Both
function orders, returned handles/record fields, aliases, snapshots and empty/
nonempty/dynamic arrays pass; this does not close #1148's distinct scalar join.

The wider unchanged assertions then expose #1151: C-interpreter writes keep
full integers or reject values outside0..255, and shared C checking permits
some known noninteger byte writes. I preserve the failed source/test/log,
retain declared byte flags, narrow integer writes modulo256 and refuse known
wrong element types before publication. I do not change the original wide
300/-1/511 assertions. Five pre-fix C source snapshots match `afca0a6a0` exactly.
I additionally reconstruct isolated old-checker/evaluator tools with the current
unchanged-interface provider closure and retain four invalid-source outputs
from six controls. These reconstructed tools are not represented as the exact
original first-failure executable images, and never replace production tools.

Dynamic production input exposes #1110 after static shadows pass: the exact
`bytes_from_string` provider returns historical ELEM_INT, so VM writes retain
full integers. I retain both producer modules and their failing actual main,
return ELEM_U8 from that exact builtin and add identity/width assertions to its
original unit. Generic FFI array conversion remains unchanged.

The complete original source then exposes #1152's development-image setup:
GCC's literal copier correctly declines the std process diagnostic's short
`\n` escape. Its required fallback searches beside the real compiler image in
`.test_output`, where the GNU-as helper is absent. I retain ordinary, verbose
and process-trace failures, explicitly configure the genuine `bin` helper and
pass complete dependency shadows plus twelve original capture/refusal methods.
No module-builder or capture-guard source changes. This setup repair is not a
claim that a passing retry retrospectively diagnoses unrelated incidents.

I also retain the unsupported `--emit-c` command, nonexistent `cast_u8` helper,
default-Clang discovery warning under Werror and missing installed-stage test
paths. I use the documented executable build interface, existing byte assignment
form and explicit `cc`. I execute the unchanged generic fixture through actual
available compiler images; its separate installed Stage1/Stage2 gate stays
open. I create no fake stage aliases, relax no warnings and remove no assertions.

## My reproduction and limits

```sh
make bin/nanoc_c nano_virt nano_vm nvm2c nvm2c-runtime \
  obj/nvm2c_service_reference obj/nvm2c_service_reference_asan
# Linux development image outside bin needs its actual assembler companion.
NANO_AS_CAPTURE_HELPER="$PWD/bin/nano_as_capture.so" \
NANOLANG_TRANSLATOR_SOURCE_COMPILER="$PWD/.test_output/ownership-source-final" \
  python3 -m unittest -v tests.test_native_byte_array_results \
    tests.test_nanolang_translator_services
```

Darwin omits the Linux helper variable and prepends
`/opt/homebrew/bin:/usr/local/bin` to PATH. I reuse the genuine C-seed-built
NanoLang development image from7cd173b30; each hash matches its prior paired
ownership receipt. Its helper/runtime closure is now explicit. This is not an
installed stage or raw fixed point. Original conversion controls use explicit
`NANO_NUMERIC_PROVIDERS=nano_virt`; the new paired gate independently executes
actual NanoLang products too. Collector source and full original commands/logs
are retained with terminal statuses and source/tool/environment identities.

Generated C fixtures/products are instrumented and link an ordinary AOT
runtime. The full evaluator/checker/VM and original suites retain their stated
ordinary/focused instrumentation; I claim no full evaluator leak freedom,
Darwin LSan, complete SDK closure or NanoLang allocation-denial equivalence.
Original C nominal allocation tests retain their actual separate authority.

Service metadata/flow/module/host admission, ordinary/private executable
verification, representation analysis, C emission in NanoLang, canonical
integration, installed/no-C-seed products, raw Stage1/Stage2 equality and all
remaining #976 requirements stay open. Parent #1110, #1142 and the release are
not closed by these component results. `retained-inputs.json` binds compressed
and raw input sizes and SHA256 values.
