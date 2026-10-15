# My nested-array and shadow-diagnostic checkpoint

Sanitizer job113754852453 at 18fd40099 reports 11 failures in 93 methods. My unchanged Darwin baseline reproduces all 11 in 293.499 seconds. Nine tests still classify supported nested-array forms as unlowerable. One invalid-array test then observes an earlier successful output. The remaining failure loses shadow source context when an unsupported foreign call is reached through deferred dependencies.

I replace the nine obsolete refusal cases with one executable fixture that checks their values and shapes through both emitters, verified NanoVM, and native ASan/UBSan/LSan:

| Context | Checked observation |
| --- | --- |
| Inferred nested local | Inner float is 1.5 |
| Nested global | Initialized inner float is 1.5 |
| Computed nested literal | Outer length is 1 |
| Integer and float results | Returned inner values are 1 and 1.5 |
| Integer and float record fields | Field inner values survive calls |
| Integer and float filled arrays | Both outer length and second-row value agree |

I retain every genuinely invalid element, result, field, arity and operand refusal. Separate invalid-case output paths prevent cascading failures when an earlier case unexpectedly publishes output. Existing prior-output preservation controls remain.

I preserve the original unsupported-shadow diagnostic assertion. The emitter now carries each queued dependency's first selected-shadow origin through transitive discovery. Initializer dependencies retain no shadow owner. On failure I report the selected target and merged line along with the underlying ABI error. A new regression checks a two-call dependency chain, selection starting at either shadow, and exclusion of the failing shadow.

My corrected Darwin gate passes all 95 methods in 308.401 seconds. My corrected Linux ARM64/GCC gate passes all 95 methods in 96.846 seconds. This component gate does not replace a final release-revision fixed-point/platform run. SQLite's separate Linux raw bootstrap acceptance remains pinned to its own receipt.
