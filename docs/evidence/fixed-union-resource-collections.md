# I reject fixed union resource-array payloads

MAC `task_e1ce4d21563d4fb3bbb998e30fc9652f` continues the affine contract after
PR411 retained my concrete payload metadata. My baseline accepted
`Owners.Some { values: array<Handle> }` in both abandoned and pass-through
signatures. The three negative baseline methods failed independently on my C
seed and both self-hosted stages; ordinary integer-array payloads executed.

I now compute resource obligations and unsupported collection payloads as
separate least fixed points. Fixed nested array annotations retain their
owning declaration and generic formal scope. Named wrappers propagate both
facts. A cycle alone creates neither fact, and an empty resource-root set
requires no further propagation. Passing an invalid union through another
function does not erase its unsupported resource-array boundary.

My C classifier walks retained `element_type` arrays and named records/unions;
my self-hosted classifier walks equivalent declared array spellings. I do not
claim new generic, tuple, row or callback substitution from this slice. Existing
resource collection, generic and owned-match guards remain in place.

## Tested checkpoint, 2026-09-17

After integrating main `8313ea81`, including C record-literal PR415:

- Fresh three-stage bootstrap passes.
- Seven paired test methods pass in 16.999 seconds: five rejection methods,
  ordinary integer and record array payload execution, and an additional C
  inline record-array assertion (22 compiler decisions).
- Forty-two adjacent generic, payload, frontend-flow and affine-boundary
  methods pass in 86.209 seconds.
- C classification checks cover deep propagation, nested arrays, module
  ownership, formal shadowing, enclosing records and resource-free cycles.
- An isolated ASan/UBSan build of that C classification executable passes
  with `ASAN_OPTIONS=detect_leaks=1`.

The ordinary record-array control exposed a separate self-hosted emission
problem. My C seed executes an inline `Values.Some { values: [Plain {...}] }`,
but both self-hosted stages select `dyn_array_push_int` for `nl_Plain` and
reject native compilation while preserving the prior artifact. The paired
classifier control uses a typed array temporary and retains the inline C
assertion. MAC `task_de0fb8219008442db8bc83e2a79eba26` and the roadmap preserve
that remaining inline-parity obligation; this is not a full inline gate pass.

Next, `task_c17b55115379414980609a5d867ccad1` implements actual selected-variant
ownership transfer. Its initial fixture is rejected by all three parsers at
`let Choice.Some { owner, label } = payload`. Qualified complete patterns,
concrete arm obligations and checked branch exits are explicit prerequisites.
I keep the owned-match guard until that implementation passes paired tests.
