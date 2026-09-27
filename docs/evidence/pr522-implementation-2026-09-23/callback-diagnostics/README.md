# I distinguish callback diagnostics from missing acceptance

My complete named-scalar-callback suite passes all eight methods with freshly
built C-seed, Stage1 and Stage2 emitters. Generated C uses Homebrew Clang with
ASan/UBSan and leak detection. I preserve original positive programs, VM
execution, indirect opcode checks and prior-output refusal checks. Three old
negative expectations now identify the actual boundaries: unresolved aggregate
globals, no exact scalar indirect target and unsupported u8 array construction.
I do not convert resource or generic callback acceptance cases into negatives.

The two unchanged fixed-resource callback methods still produce four failures,
one in each native stage per method, while the C seed passes. My source owner
emitter has no function parameter tag and emits only direct calls. The affine
value-call graph explicitly rejects CALL_INDIRECT, and owned native emission
has no indirect branch. General native indirect translation also restricts its
result candidates to scalar tags. A source spelling exception would not supply
target identity, verified ownership transfers or native cleanup. These remain
required implementation work under task_ed8e4810c6554f3c9dd73def105ca6a1.

## I reject an unqualified alias-scan optimization

A fresh three-second sample of the pre-experiment instrumented shadow child
records 734 top-of-stack samples in env_define_var_with_type_info and 234 in
env_set_var. I experiment with recent-first existential string-alias scans.
The retained patch is experimental evidence, not a production change.

An isolated benchmark changes only env.c between builds, retaining identical
other compiler objects. It inserts 20,000 aliases after 2,048 integer bindings.
Three alternating runs have median insertion times of 0.271074 seconds before
and 0.010051 seconds after. The benchmark checks all GC objects are released.
This workload favors recent aliases and does not represent the full compiler.

The prior fully instrumented seed at 9964329ba passes all 898 compiler shadows
under the unchanged 60-second deadline (141.05 seconds total build wall time).
The experimental fully instrumented current-source seed reaches the same
shadow deadline and fails (165.95 seconds total). These whole-build timings
include native/module compilation; they are not shadow-only timings. Their
source checkpoints and concurrent host work differ, so they do not isolate a
regression cause or establish Linux timeout causality. I retain the initial
invalid `-v` invocation separately from the corrected `--verbose` run.

I revert the production scan change. Ordinary environment/interpreter gates,
fresh ordinary bootstrap and 48 environment assertions pass during the
experiment; the same 48 assertions pass with the restored production source.
The new lifetime controls cover early, recent and shadowed aliases, replacement
of original bindings and final release. A fully instrumented run of those 48
assertions also passes with leak detection disabled as in CI; the explicit GC
object assertion checks the added fixture's cleanup.

## I preserve new hosted failures

Run 35934003714 at 9964329ba remains unqualified. Units-02 repeats the unchanged
60-second bootstrap-shadow timeout. Units-00 rejects the existing stage-2
function-reference purity case (task_fa1064830d1b43d3a7d8753d03db76a6). The macOS
arm64 scalar reconstruction gate has 18 failures because generated equality
conditions trigger -Wparentheses-equality under -Werror
(task_e88279df132f4f6ea092b7a288c9cf88). I retain the actual logs and require
repairs without dropping warning flags or accepted programs.

Complete callback acceptance, final-source fixed points, hosted qualification
and release documentation remain required. #522 stays draft.
