# My portable nonfinite formatting policy

I track task_e92a45b66a104e9ba3854cd5f994df8b after the exact f736 Darwin
managed gate failed its nineteenth method at negative quiet NaN index28.
My retained reports live on PR739, commitdd76142a. A fresh six-value native
probe observes that Linux libc retains negative NaN signs and Darwin libc omits them
for both quiet and signaling payloads. Both spell signed infinities identically.
This is a formatting-policy difference; I do not relabel the failed gate a pass.

I make my already implemented managed-core sign policy explicit across hosts:
positive NaN is `nan`, negative NaN is `-nan`, infinities are `inf` and `-inf`.
I inspect the sign/exponent/significand as integer bits before any floating
comparison or libc formatting. Payloads and quiet/signaling bits remain unchanged
in stored and transported values. This selects text, not arithmetic normalization.
Finite conversion retains six-significant-digit C-locale nearest-even `%g` rules,
including signed zero. Finite VM print retains its existing integral `.1f` rule.

My first checkpoint provides a shared C99/C11 nonfinite formatting helper and
mechanically identical generated source, then uses it in VM string conversion,
value display and native generated C conversion/printing. My managed integer
formatter already implements the chosen nonfinite spelling and stays unchanged.
VM print must test its finite range before casting to an integer; I track that
statically found undefined-conversion path separately as e48563e1f62c4bd89648596c8c55849b.
All ownership/error handling stays unchanged.

My reference corpus will distinguish independent finite libc references from
explicit nonfinite policy expectations selected by integer bits. It keeps all
2077 values, exact lengths/bytes, allocation controls and sanitizers. Native
reference observations remain separately asserted/recorded for both hosts;
I do not drop NaNs or accept either spelling. Generated tests must construct
nonfinite values through integer bit transport, because Darwin `%a` itself
loses negative NaN signs. Literal transport is not an acceptable oracle here.

Before closure I require fresh exact-bit VM/native/managed LLVM/Wasm conversion
and printing controls on Linux and Darwin, both quiet/signaling signs, infinities,
zero signs, finite rounding boundaries and randomized corpus, plus the full71
managed methods. I preserve prior failed artifacts and freeze each new source,
tool and harness pin. I qualify the VM print correction under float-cast-overflow
sanitization, with extreme/nonfinite inputs, after production review.

Legacy interpreter/array/formatting helpers, paired legacy C source emitters and
public C target are explicit subsequent checkpoints under e92. They must consume
the same signed nonfinite policy before that parent closes. The public C-target
agent owns its emitter; I coordinate its provider integration. I do not change
source-literal metadata, tracing/debugger diagnostics or arbitrary user printf
formats as an accidental part of this scalar conversion checkpoint.

## My legacy scalar checkpoint

I track local child_df2b1ff3d75d4b43b2471696ee0b613e after merged PR745.
I extend the shared formatting header with a stream helper that selects signed
nonfinite text from integer bits and otherwise retains `%g`. My interpreter
uses it for scalar and array printing and uses the existing bounded buffer
helper for casts, generic value-to-string, float-to-string and the existing
`%g` format substitution path. I preserve each route's existing finite `.0`
suffix rule, buffer bounds, allocation and string ownership. Arbitrary user
format specifications, source literals and debugger/tracing output remain
outside this conversion slice.

My C-seed legacy runtime embeds the same shared helper source. My selfhost legacy
runtime receives an exact generated provider from the same header, with mandatory
shadow checks and regeneration identity verification. Both existing print and
float-to-string helpers consume the common policy. I do not fork its logic into
the emitted runtime strings or replace any user expression with multiple reads.

Before closure I require a fresh bootstrap, exact-bit interpreter/C-seed legacy/
Stage1/Stage2 conversion and captured print controls, qNaN/sNaN signs, infinities,
zero signs and finite boundaries, scalar/array/format routes where each emitter
supports them, unchanged input bits, and adjacent conversion/bit-transport gates.
I retain and separately report unsupported source routes rather than treating
them as a pass. Public C-target work remains with its assigned agent and must
consume this provider in a separately reviewed change. Parent e92 stays open
until all required paths and both host qualifications are complete.

My first fully bootstrapped scalar fixture retained two finite-output expectation
failures: Stage1 and Stage2 print float identifiers through their existing
float-to-string conversion, retaining `.0`, whereas my interpreter and C-seed
print use `%g`. I verified this lowering in the unchanged base before correcting
the route-specific fixture. All signed nonfinite observations and provider
identity passed in that run; I retain its failed status and qualify fresh outputs.

My generated C-seed array print already routes through `nl_to_string_array`,
whose float elements use `float_to_string` and retain `.0`; my interpreter
array print uses `%g`. I retain the first array fixture's finite-only mismatch
and preserve these existing route-specific expectations. My Darwin bootstrap
passes independently; its first fixture setup lacks the separate `bin/nano`
target, so I build that interpreter explicitly before freezing the complete toolset.
