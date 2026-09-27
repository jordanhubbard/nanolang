# My C scalar-global source checkpoint

My C owned producer now retains explicit scalar globals in source order,
emits their initializers before main or the synthetic shadow entry, and emits
loads and checked mutable stores with exact slot tags. Locals and parameters
retain precedence over globals. The public verifier checks initialization and
helper-call preconditions. I serialize extension kind 3 in ownership format 4,
with or without ordinary/resource union facts. Managed-array and borrowed-helper
profiles remain separate.

The original `test_scrutinee_call_evaluates_once` source passes unchanged through
mandatory shadows, verified bytecode, VM execution and generated native C with
ASan/UBSan and leak detection. Seven additional accepted sources cover ordered
initializers, retained/overwritten strings, Boolean and float mutation, local
and parameter shadowing, a loop counter and an ordinary union beside a resource.
Nine refusals preserve prior output: untyped globals, immutable and wrong-tag
assignments, owner/aggregate globals, duplicate declarations, early reads,
helper calls before initialization and a failing effect assertion in a shadow.
Global annotations remain explicit under the existing source checker; local
inference does not establish global inference.

All 17 cases pass scoped codegen/checker ASan/UBSan with leak detection; linked
dependencies remain ordinary objects. Both existing owned-union source methods
also pass, retaining the 13 accepted/nine refused corpus across both producers
and lexical match-arm names. This focused regression does not replace the full
source-borrow gate or release qualification.

I retain intermediate failures: a misplaced code insertion stopped compilation;
an unset Python Make variable accidentally ignored command failure; direct
fixture execution exposed a missing compiler-method receiver and the unsupported
untyped-global assumption. I corrected those fixture/build issues, then ran the
actual target successfully. The earlier zero Make exit is not test evidence.

`instrument_c_frontend.py` consumes the retained build2 log after decompression
and runs with `PYTHONPATH=.`. Source and terminal hashes are in `provenance.json`
and `logs.json`. The self-hosted producer still refuses global initializers;
fresh native stages and the original three-compiler acceptance remain required.
PR #522 stays draft.
