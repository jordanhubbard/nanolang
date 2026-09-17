# My guarded passive scalar inputs

I accept version-2 scalar input records only after checking an executable
function-entry guard prefix. Each recorded external input has a matching
`LOAD_LOCAL`, `TYPE_CHECK`, `ASSERT` sequence for its declared int, bool, string,
or float tag. The prefix runs before any branch. Every call starts at function
entry; I forbid writes to recorded input slots throughout the function, so a
later branch cannot invalidate the checked value. Ordinary module verification
still checks stack shapes and branch boundaries.

I retain version-1 zero-read behavior. The record layout and section feature bit
are unchanged; the payload version makes old readers refuse the extension. I
confirmed that the preceding version-1 reader rejects a saved version-2 module.
Both versions retain exact canonical text/binary roundtrips.

My focused gate runs six methods: the existing metadata/execution matrix,
four scalar inputs in one `par` graph, an external input feeding a `flow`
dependency, an ordinary failed runtime tag assertion, old/unknown versions,
and incomplete or inconsistent guard claims. The positive graph prints
`42`, `true`, `guarded`, and `1.5`; the flow prints `43`. Both NanoVM and strict
native output agree. Canonical text roundtrips preserve complete module bytes.
My retained version-1 matrix passes 249 checks; the adjacent verifier passes
96 cases and canonical disassembly passes 210 checks.

An initial test incorrectly demanded native translation refusal for a boolean
argument passed to a guarded integer parameter. Inspection showed that native
inference preserves the boolean and emits a false integer-tag check. The
corrected test requires both executions to stop at the assertion before node
output; no native source repair was needed. I cancelled the suspected defect
`task_f7ed4e430c9b41458b4ca9123671765d` as not applicable and retained its history.

This completes only `task_b26c733ca21841768b4dd6c67e7ed094`. The parent
`task_bf571298c10d4cc5a387b9f233ff3c40` still includes broader immutable-input
proof. U8, aggregates, captures and transitive call proofs remain outside this
record. Neither frontend emits `par`/`flow` eligibility records yet. Trusted
foreign intrinsic identity remains `task_20f6cb36fbf24bba987b4ea503529438`.
