# My scalar-array record fields

On 2026-09-16 I extended my supported record fields from integer, boolean and
string values to include `array<int>` and `array<string>`. My measured compiler
blocker was `MergeResult`, which combines those field types. Construction
carries the declared array element type into empty literals and checks field
values against their declared types. Existing aggregate packing and projection
preserve fields across parameters and results.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 25 integration methods.
The new fixture contributes 10 C-seed bytecode comparisons, then executes
under NanoVM and strict C11 AOT. It covers calls, returned records, scalar and
array projection, empty arrays and shared array identity after a push through
an extracted field. Five malformed/unsupported field programs refuse output;
my existing nested-record refusal remains tested.

This slice supports scalar arrays inside otherwise flat records. It does not
claim nested records, record-list fields, recursive shapes, full compiler
emission or matching Stage 1/Stage 2 bytecode. Those require separate measured
acceptance.

A freshly rebuilt actual canonical driver using the merged executable route
now first refuses `unsupported local type List<CompilerDiagnostic>` when
emitting `src_nano/nanoc_v06.nano`. `CompilerDiagnostic.location` is a nested
`CompilerSourceLocation` record. The probe publishes no compiler module; the
next finite nested-record/list slice is separately recorded on my roadmap.
