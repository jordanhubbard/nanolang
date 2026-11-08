# Completed STRING constructors

I measure the alias scans in my C interpreter's environment before changing them.
My paired `baseline.json.gz` and `constructor.json.gz` retain the actual source,
header, unit, commands, stable input hashes and stable native product hashes.
Each cohort performs 10,000 fresh declarations and 10,000 fresh replacements.

| Live outer bindings | Baseline declarations / replacements (CPU seconds) | Constructor path declarations / replacements |
| --- | --- | --- |
| 32 | 0.001873 / 0.001289 | 0.002224 / 0.001208 |
| 512 | 0.004020 / 0.005823 | 0.001219 / 0.000940 |
| 4096 | 0.041082 / 0.028081 | 0.001356 / 0.000965 |

These are single paired measurements, not whole-compiler timings. The smallest
declaration cohort becomes slower. The larger cohorts expose the redundant scan.

I elide that scan only when the value just evaluated from the exact expression
is a STRING from a literal or a two-operand primitive `+`, without return, break
or continue flags. Both evaluator paths finish by constructing a fresh STRING.
My generic APIs, identifier and call results retain their existing alias scans.
I do not infer ownership from a reference count, function spelling or a cache.

`environment-unit.json.gz` records my complete environment unit built from the
changed sources under ASan/UBSan. It passes borrowed identifier/call aliases,
fresh transfers, metadata, return propagation, malformed unary fallback,
same-pointer assignment and the existing record/static-array controls. Strict
C11 syntax checks with warnings as errors also pass for the environment,
evaluator and environment unit.

I extend my existing language STRING contract with literal and primitive `+`
bindings, replacements, borrowed function results and aliases retained across
replacement. `installed-seed-language.json.gz` records a successful syntax and
native run with my installed seed. That older executable does not qualify the
changed evaluator. My hosted fresh-provider selection, platform gates and
original deadlines remain required. I have not closed #1065 or qualified a
release with these measurements.
