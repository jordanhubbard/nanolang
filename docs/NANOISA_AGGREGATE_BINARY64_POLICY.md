# My existing floating-array arithmetic policy

I execute child task_bd474b339063ea2fe3915b29f12f9c80 under aggregate
parent task_3717eda847a74122916571444478ee0f from main a5985842. This contract
precedes production. Scalar callback specialization belongs to d099/5009 and
continues independently; I do not count callback coverage as array operators.

I apply my already selected scalar binary64 ADD/SUB/MUL/DIV policy separately
to each existing FLOAT array element operation. A NaN arithmetic result becomes
positive quiet bits7ff8000000000000; either signed-zero divisor produces positive
zero, including a NaN numerator. Ordinary finite operations retain per-operation
binary64 nearest-even rounding, without contraction or fast-math. I preserve
transported input payloads, negation, element order and independent snapshots.
I consume unchanged shared arithmetic helpers rather than copy their policy.

My first checkpoint covers existing VM generic numeric array/array and scalar
broadcast execution; interpreter dynamic and legacy array branches; generated
C runtime array/broadcast helpers; and both C-seed and selfhost legacy emitted
element loops. Existing VM int-to-float promotion stays as it is. I do not widen
source type compatibility, shape rules, mixed-tag admission or empty-array
metadata. Modulo, integer overflow/division, external math and new native/LLVM/
Wasm aggregate admission remain separately explicit; scalar helper gates do not
establish those missing target contracts.

I preserve existing allocation/alias ownership, result construction, shape and
length checks, failure cleanup and nested recursion. Empty arrays perform no
arithmetic. Nested interpreter/C helper recursion consumes the same floating
leaf helper without claiming additional canonical nested-array admission.
Generated string-operation sections contain array helpers, so I publish the
existing guarded arithmetic provider before those helpers as well as the math
section; either section order must compile without implicit declarations.

My source-order audit also records task_30c04e6d0ef594ef20bf7f9fec161969:
both legacy producers currently snapshot the right-hand array before the left
scalar in scalar-left broadcast. I correct ordered temporaries so left evaluates
once before right, then capture length after both operands. Known typed array
and scalar snapshots retain their existing types. The C-seed runtime-helper
fallback must preserve the same source order without inferring an unknown type
as INT; its existing GNU legacy expression boundary remains explicit. I keep
public C target portability separate. No old failed source/artifact is executed.

Before production qualification I send an independently reviewable checkpoint.
I freeze new ordinary fixtures and tools, then require actual interpreter,
C-seed legacy and Stage1/Stage2 legacy output, paired canonical VM observations,
exact generated helper inspection and direct VM allocation/cleanup controls.
Every admitted native route uses unchanged strict sanitizers; unsupported native/
LLVM/Wasm aggregate routes remain explicit refused/unimplemented obligations,
not passing matrix cells. I require new full bootstrap where selfhost sources
change and record its exact pin independently from later integration.

My observations use integer bit patterns for both signs of quiet/signaling NaNs,
infinities, signed zeros, finite rounding boundaries, overflow and subnormals.
I check all four operations, array pairs, both broadcast directions, unchanged
input bits/aliases, empty inputs and ordered side effects, including an operand
that grows an alias before length capture. Neighbor int/string semantics and
mismatched shapes retain their established behavior. Independent expected bits
and previously qualified scalar results cross-check element results; host libc
NaN spelling is not an arithmetic oracle. Linux and Darwin claims remain tied
to their actual frozen runs. Full3717 and release remain open until all required
route/shape/target acceptance is measured; this checkpoint alone closes neither.

## My paired source typing prerequisite

I record task_90b473627cffdb98e69d5a5641937d30 before corrective code. The first
frozen3087067f four-method gate passes C-seed/interpreter observations, but all
eight Stage1/Stage2 legacy/canonical subcases refuse FLOAT broadcasts because
check_binary_op chooses scalar FLOAT before recognizing an array. I retain all
logs and unchanged source/tool hashes. Static follow-up also finds canonical
binary emission chooses F64 operations when one operand is FLOAT and typed I64
otherwise, without an array result path. I do not execute such invalid outputs.

I add an exact resolved array arithmetic classifier before scalar promotion,
limited to existing flat array<int>/INT and array<float>/FLOAT operands. Both
arrays must have identical known element kind; one scalar must match that kind.
The result preserves the array type. Four arithmetic operators reuse existing
VM generic ADD/SUB/MUL/DIV execution; no typed scalar opcode consumes an array.
Unknown, mixed element/scalar types, nested/nominal/bool/string arrays and modulo
remain outside this new paired source slice and receive a checked refusal where
this path is selected. I preserve comparison handling and scalar compatibility.
I do not infer a missing element kind from an unrelated scalar.

The checker reports incompatible array arithmetic before publication rather than
letting an error-UNKNOWN pass an annotated let. Canonical expression type and
emission use the same exact resolved profile, and legacy emission retains its
already reviewed snapshots/helpers. Mandatory helper shadows, explicit negative
publication preservation and the unchanged positive arithmetic/order fixtures
precede closure. A fresh corrected bootstrap is required. Integer neighbors
establish source dispatch/order only; integer overflow policy is unchanged.

I narrow the shared-checker selection after independent review: it changes only
four-operation expressions containing a resolved flat numeric array operand.
Within that selection I reject a mismatched scalar or array element kind. Other
legacy array profiles retain their prior checker result, including string,
nested and modulo expressions; I do not claim new support for them. Canonical
emission independently refuses arithmetic arrays outside the exact numeric
profile before choosing any scalar opcode. This keeps the new paired profile
from silently rewriting unrelated legacy checker behavior.

## My legacy direct-call result prerequisite

I record task_6ed0cd1295af4a42b04133ddc6d39d98 before repair. Corrected985321f7
passes exact arithmetic across all producers and canonical ordered broadcasts,
but both legacy selfhost stages reject generated C for the unchanged ordered
fixture. Legacy array selection consults identifiers alone and drops the known
array<float> return type of source(values). I preserve the refused C and logs.

I extend that selection only for direct unbound declared calls with exact
array<int>/array<float> results. Existing identifier handling stays intact.
Lexical bindings take precedence; I do not treat a same-named declaration as
proof for a bound callable or guess UNKNOWN. I reuse existing resolution and
array helpers, including their reviewed operand snapshots. Other expression
shapes and callback admission remain unchanged. Mandatory helper shadows and
the unchanged source-order fixture precede closure after a fresh bootstrap.
