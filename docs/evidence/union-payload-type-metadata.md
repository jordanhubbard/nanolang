# I preserve concrete union payload metadata

I continue the affine prerequisite tracked by
`task_1f64c9b88a5248dcbda2258dcbee99f7`. I retain parsed payload annotations in
independently owned AST, environment and module metadata trees. My nominal
binding preserves declaration formals inside arrays and generic parameters.
My supported generic substitution recurses only through `element_type` and
`type_params`, replacing complete argument trees there. Copying tuple, row and
function metadata does not substitute formal names embedded in those shapes.
I do not claim those substitutions, owned collection transfer or selected-variant
ownership transfer.

On 2026-09-17, my C baseline at merged PR388 rejected ordinary `array<T>` and
`array<array<T>>` match projections as non-arrays. After metadata preservation,
my C seed executes string array payloads, imported payloads and a formal `T`
that shadows a resource record. It retains rejection of `Box<Handle>`.
My fresh three-stage bootstrap and five methods on each self-hosted stage
passed. My module metadata gate includes a lifetime test that frees the parsed
AST and registering environment before inspecting the retained payload copy;
concrete substitution leaves the declaration unchanged.

My initial nested-array native execution met the independently tracked C
literal emitter defect `task_c2ebfd28c24345daaa8c31dac75b45ac`. A typed temporary
reproduced the same `DynArray **` initialization error. After integrating
PR405 and the v5.1.0 contract through `6a68f981`, my fresh bootstrap, module
metadata gate, five paired methods (15 compiler decisions, 31.264 seconds),
and typechecker unit tests pass. I preserve the initial failures as dependency
evidence; they no longer describe the integrated result.

My isolated ASan/UBSan metadata execution passes with `detect_leaks=0`.
LeakSanitizer separately reports 245 bytes in legacy registered field-name,
formal-parameter and import-tracker allocations, tracked by
`task_00c47a5d65d04c48914864ec0de553d6`. I do not claim leak freedom.
My initial instrumented build at `-O1` met an existing format-truncation
warning; a separate `-O0` object directory built successfully. The aggregate
metadata target also invokes a NanoVirt fixture with independent legacy leaks;
I ran the metadata executable directly for the scoped result.

I also observed that the fixed declaration
`Owners.Some { values: array<Handle> }` accepts `abandon(Owners)` in the C seed.
That classification gap is tracked separately by
`task_e1ce4d21563d4fb3bbb998e30fc9652f`; generic rejection does not prove that
fixed nested resource fields are guarded. I keep both ownership obligations
open and the full roadmap publication hold in place.

My final integrated source `51ec1cd9` includes PR398 and PR410 through main
`48bc99e0`. Fresh bootstrap, module metadata and five paired payload methods
pass (31.700 seconds), as do sixteen existing generic methods across all three
frontends (48 decisions, 55.612 seconds). The integrated ASan/UBSan metadata
executable also passes with leak detection explicitly disabled. These are
prerequisite gates, not full affine or release acceptance.

A separate boundary probe uses `Bundle<T>.Some { value: (T,int) }` with
`Bundle<Handle>`. My C seed rejects it with an ownership diagnostic. Both
self-hosted stages emit no executable, but fail later on generated
`Tuple_T_int`; I do not misdescribe that as an intentional ownership diagnostic.
MAC `task_bcd773ad3c084ce099a3da5aef682fef` tracks early rejection or complete
substitution/lowering. I retain the existing generic-array rejection checks and
do not remove diagnostics to admit these unsupported shapes.
