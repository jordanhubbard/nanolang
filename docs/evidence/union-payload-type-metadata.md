# I preserve concrete union payload metadata

I continue the affine prerequisite tracked by
`task_1f64c9b88a5248dcbda2258dcbee99f7`. I retain parsed payload annotations in
independently owned AST, environment and module metadata trees. My nominal
binding preserves declaration formals inside arrays and generic parameters.
My supported generic substitution replaces complete argument trees inside
array/generic payloads; this does not establish tuple, row or callback generic
substitution, owned collection transfer or selected-variant ownership transfer.

On 2026-09-17, my C baseline at merged PR388 rejected ordinary `array<T>` and
`array<array<T>>` match projections as non-arrays. After metadata preservation,
my C seed executes string array payloads, imported payloads and a formal `T`
that shadows a resource record. It retains rejection of `Box<Handle>`.
My fresh three-stage bootstrap and five methods on each self-hosted stage
passed. My module metadata gate includes a lifetime test that frees the parsed
AST and registering environment before inspecting the retained payload copy;
concrete substitution leaves the declaration unchanged.

My nested-array native execution still meets the independently tracked C
literal emitter defect `task_c2eb` and requires the peer repair before the full
paired gate can pass. A typed temporary reproduces the same `DynArray **`
initialization error; I preserve both logs instead of claiming success.

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
