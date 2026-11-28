# I trace live assignment values

I qualify source `e542ae38f2f68b19afb3b491df568905b19a53f3` under #1208.
I serialize old/new assignment values synchronously before replacement retires
old storage. I add no snapshot allocation or public layout change. My ordinary
non-tracing assignment path keeps its existing ownership behavior.

My actual tracing-enabled build also exposes an incompatible enum conversion
in function argument tracing: runtime ValueType is passed to a declaration
Type formatter. Their ordinals differ. I use a private runtime-kind formatter
and check the actual output labels for all14 runtime kinds, preserving strict
warnings and the existing output structure.

My focused native test observes21 live old/new values across enabled, filtered
and disabled tracing, including string/callable/record replacement, scalar
updates and self-assignment. The observer inspects actual old callable and
record metadata, then invokes the production tracer. The output has seven
unfiltered assignment events, one number-filtered event and no disabled event.
All14 runtime labels are checked against production function trace output.
The complete fixture passes normal and fresh complete ASan/UBSan providers
with leak detection enabled; its fresh instrumented build/run takes72.898s.

My private negative evaluator moves only the retirement call ahead of
serialization while keeping the same observed old Value. The identical native
test compiles and reports heap-use-after-free in nano_test_trace_var_set while
reading the old string. I retain that source, object, executable and full
sanitizer report. My reader reconstructs this deliberate ordering mutation
from the candidate source; I do not present it as a second production pin.

All134 original interpreter and88 checker methods pass normally and with
complete ASan/UBSan providers, retaining detect_leaks=0 for those broad suites.
The original134-method suite also passes with a tracing-enabled evaluator
calling production tracing directly, without the observation rename; tracing
configuration is disabled for that broad compatibility suite. The focused
fixture exercises the enabled and filtered configurations.

My original strict nine-route/64-call audit still retains zero names. The
complete expression/globals fixture,991/991 record-binding regression,
borrowed-value and environment controls pass. My complete record/union fixture
still reports3720 leaked bytes in159 allocations after successful assertions
and shadows. General #1116/#1140 reclamation and paired current-source release
acceptance remain open.

I retain three build refusals: the original undeclared-old_value failure,
my first invalid Make-rule insertion, and the strict tracer enum-conversion
failure. The corrected Make insertion preserves its original dependency line.
I change no assertion, warning flag, deadline, source permission or Puck-owned
backend/capture/checkpoint/CI input, and assign no historical timeout cause.

I verify my lossless archive with:

```sh
python3 docs/evidence/tracing-assignment-linux-20261123/read_archive.py
```

My reader passes1499 records and1388 exact Git inputs. It verifies compressed
and raw hashes, unchanged original fixture and tool identities, live/output
observations, the exact late-serialization mutation, expected failures and all
native products. Provider object sizes/hashes and complete linker inputs are
recorded in qualification.json; focused and unwrapped evaluator products are
retained losslessly. Identical objects are shared with the previous
record-binding archive, whose manifest is hash-bound. Source, observers,
commands, streams and original private paths remain preserved.

All old roots, dirty audits/stashes and NAS/TCC/migration pauses remain intact.
I perform no NAS transport, probe, cutover or deletion and claim no Darwin or
whole-release acceptance.
