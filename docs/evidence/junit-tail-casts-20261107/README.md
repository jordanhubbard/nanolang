# My direct string-cast return gap

I retain the original combined-tree JUnit failures under #1081. Return emission
recognizes string binary builtins but misses the existing `to_string`,
`cast_string` and `bool_to_string` cast aliases. Ordinary expression emission
already lowers these names to `CAST_STRING`. I route undeclared aliases through
that same checked call path before `RET`, preserving the declaration guard and
actual function tail calls. My mandatory emitter shadow checks the cast/return
sequence and the structural declared-call distinction before separate frontend
name admission.

I add executed original-producer/VM/native controls for all three aliases,
single-expression integer interpolation, nonintegral float conversion, once-only
argument evaluation and wrong-arity output preservation. My override test uses
`bool_to_string`, which the existing C seed accepts as a declared function.
I do not change frontend rules that refuse declarations named `to_string` or
`cast_string`, or modify the original complete f-string fixture/assertions.

My [bounded existing-seed probe](existing-seed-fstring-baseline.json) independently
fails the original full fixture's zero-float shadow at line 33 before publication.
These are existing local binaries with recorded hashes, not fresh candidate
providers. My [first local native counterpart](first-local-counterpart.log)
also cannot complete because `bin/nano_aot_runtime.o` is absent. That initial run
retains my incorrect positive-override assumptions for the two prohibited names,
and the existing seed's acceptance of `(str_trim 1)` where the self-hosted test
requires refusal. I correct only my newly introduced override test premise; I
keep the existing trim assertion and missing native qualification visible.

My [report/storage controls](report-storage-controls.log) pass 14 real filesystem,
child-process and XML consistency tests. They do not qualify the compiler repair.
The branch-scoped hosted diagnostic freshly bootstraps all providers, executes
every canonical string method through seed and both stages, and then runs the
complete original JUnit selection through C-native seed, seed NanoVM and both
self-hosted products. It preserves full reports, all failed outputs and source/
provider identities. Failed focused tests do not suppress the full diagnostic;
failed routes remain failures while later routes are collected. I keep #1081 and
both roadmap acceptance rows open until actual native and full-corpus evidence.
