# My fresh record binding experiment

I compare exact `src/env.c` from `93474a7bbd6405259a83218997ba764f0b9f808b`
with a private candidate. I leave production sources unchanged. The candidate
transfers only a completed `AST_STRUCT_LITERAL` result of kind `VAL_STRUCT`
into `env_define_evaluated_var`. Other expressions keep their record copy;
borrowed types keep their existing identity. Ordinary binding and assignment
APIs keep their original behavior. I do not infer callback argument ownership
from an inherited source location.

My actual native controls exercise detached Unicode names, STRING snapshots,
empty records, nested records with borrowed array fields, copied identifier,
call and unknown-expression results, both borrow types and all three propagated
control-flow flags. These are direct environment API controls with explicitly
constructed AST descriptors. They do not execute the evaluator, ordinary
function calls, source-generated programs or foreign records.

Each timed execution constructs, binds and explicitly destroys 100,000
four-field records. The old form creates a literal and a binding copy; the
candidate binds the fresh literal. Both produce checksum400000. The harness
explicitly destroys both old carriers, unlike current ordinary frame cleanup.
This measures construction/binding cost with accounted experiment cleanup;
it does not measure current frame retention or solve #1116.

| Observation | Current copied binding | Private fresh binding |
| --- | --- | --- |
| Median process CPU, twenty alternating pairs | 0.012770s | 0.0078135s |
| Direct binding malloc calls, separate counted image | 300000 | 0 |
| ASan/UBSan native control and 100000 bindings | pass | pass |

I retain all forty plain samples and both counted and sanitized executions,
exact sources, compiler commands/version/driver hash, streams and before/after
source/image hashes in `measurement.json.gz`. All six actual Mach-O images
remain losslessly compressed under the report's local output path. Managed
usage/object counts match before and after each timed loop. This is not leak
sanitizer qualification: the existing Darwin runtime uses `detect_leaks=0`.
I do not establish complete native SDK/header/linker closure.

I retain two initial mechanical-edit refusals. My first replacement text also
matches an assignment helper, then the freshness call also matches assignment.
Both refuse before any compilation. I scope all replacements to the complete
local-binding block and require each replacement exactly once.

My first completed native plan passes all44 executions and stable-input/image
checks but fails its final32MiB storage guard: six uncompressed native images
occupy49.5451MiB. I retain that refusal and its exact collector source. I do not
raise the cap. My corrected collector builds one original/candidate pair at a
time, checks storage after every command, reserves10MiB before compilation and
losslessly compresses each verified image after its final execution. The same
32MiB guard passes with a17.1701MiB observed peak. No source factory is launched
in either experiment. Both plans retain their complete measurements.

I require production integration, meaningful alias/escape/evaluator controls,
original whole-source alternating comparisons and unchanged Linux/Apple gates
before adoption. The measured binding saving is not a whole-compiler gain,
reliable source acceptance or release readiness.
