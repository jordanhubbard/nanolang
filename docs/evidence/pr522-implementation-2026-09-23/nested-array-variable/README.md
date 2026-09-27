# My nested record-array variable evidence

Hosted run `35927493387` units-06 rejects a correctly typed `array<array<Plain>>`
variable supplied to `Box<array<Plain>>.Some`. My earlier match-result evidence
retains the hosted terminal and its local reproduction at `5a179473c`.

The temporary diagnostic in `variable-debug.log` shows equal nested element
kinds and record names, but the variable retains `Plain` as a redundant name on
its inner array node. The substituted annotation does not. My payload check now
uses the existing exact recursive type comparator for array annotations. It
still requires matching element kinds and nominal declarations. I remove the
temporary diagnostic from production code.

`variable-expanded.log` passes all three owning methods: the original native
positive corpus, eight wrong-type prior-output refusals through both C producers,
and a new bytecode case that selects and reads the nested record value.
`variable-instrumented.log` passes the eight refusals and the bytecode value case
with the checker translation unit instrumented by Homebrew LLVM ASan/UBSan,
with leak detection enabled. Other linked compiler objects are ordinary objects.
The instrumented build log records this scope; I do not claim whole-compiler
instrumentation. Compressed logs preserve exact uncompressed bytes and hashes.

MAC `task_2470ce5099c24aacb031b1d881f9804e` tracks this repair. Complete hosted
sanitizer/platform qualification and final-source fixed points remain required.

All ten paired canonical match methods pass in `variable-match-regression.log`.
The self-hosted production source is unchanged; this run uses the retained
shadow emitter and current C producer.

The broader 42-method adjacent run retains 21 failures: four native-stage
resource callback shadow failures, fourteen native-stage generic function-value
translation failures and three named scalar callback refusal-diagnostic
mismatches. These remain acceptance blockers. I do not discard them because
the C-only controls pass. Separate MAC tasks and roadmap entries track them.

For the compiler-shadow investigation I build every C compiler object with
Homebrew LLVM, ASan/UBSan and `-O0`, retaining ordinary external libraries.
The first attempt infers the wrong repository root from the temporary executable
location and fails module compilation. After placing a separate executable in
`bin/`, the run reaches interpreted shadows and ends with the generic failed
shadow diagnostic, without an explicit timeout or sanitizer finding. I retain
both terminals. The sampled child spends 519 of 1,272 top-of-stack samples in
`env_define_var_with_type_info`, 203 in `env_get_function` and 166 in `env_set_var`.
This identifies investigation targets; it does not establish the failure cause.
The run keeps the 60-second deadline and uses CI's `detect_leaks=0`; it is not
leak qualification. Docker is unavailable on this host.

I retain job metadata before requesting cancellation of superseded hosted runs
`35927493387` and `35931004648` to free workers for the current candidate.
Cancelled or incomplete checks do not count as passed.

The C-only adjacent gate passes all 34 nominal record-array, resource callback
and generic function-value methods in `variable-c-adjacent.log`.
