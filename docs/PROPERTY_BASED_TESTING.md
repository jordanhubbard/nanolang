# Property-Based Testing (proptest) for NanoLang

NanoLang includes a lightweight, deterministic property-based testing helper module: `modules/proptest/proptest.nano`.

This is the **single canonical doc** for:
- Writing properties and generators in canonical NanoLang
- Understanding shrinking and counterexamples
- Reproducing failures deterministically (seeded)
- Running in CI

## Mental model

A property test is just code that:
- generates inputs
- checks a property
- on failure, shrinks to a simpler counterexample
- reports a `PropertyReport` you can assert on in a normal `shadow` test

There is **no special syntax**: properties are plain functions returning encoded outcomes.

## Outcomes (pass / fail / discard)

Your property function returns an encoded `string` outcome:
- `(proptest.prop_pass)` — case passes
- `(proptest.prop_fail "message")` — case fails (will trigger shrinking)
- `(proptest.prop_discard "reason")` — precondition not met (runner tries another case)

Use **discard** for preconditions (e.g., division by zero), not for “random retries”.

## Quickstart: a unary int property

```nano
from "modules/proptest/proptest.nano" import int_range, forall_int, report_passed, report_summary, prop_pass, prop_fail

fn prop_non_negative(x: int) -> string {
    if (>= x 0) {
        return (prop_pass)
    } else {
        return (prop_fail "negative")
    }
}

shadow prop_non_negative {
    let gen: proptest.IntRangeGenerator = (int_range 0 100)
    let report: proptest.PropertyReport = (forall_int "non_negative" gen prop_non_negative)
    assert (report_passed report)
    (println (report_summary report))
}
```

## What exists today (API aligned to current implementation)

### Generators

- `int_range(min: int, max: int) -> IntRangeGenerator`
- `int_pair(first: IntRangeGenerator, second: IntRangeGenerator) -> IntPairGenerator`
- `int_array(element: IntRangeGenerator, max_length: int) -> IntArrayGenerator`

### Configuration (trials / shrink / discards / seed)

- `config_default() -> RunConfig`
- `config(trials: int, max_shrink_steps: int, discard_limit: int, seed: int) -> RunConfig`

### Runners

- `forall_int(name: string, gen: IntRangeGenerator, prop: fn(int) -> string) -> PropertyReport`
- `forall_int_with_config(name: string, gen: IntRangeGenerator, prop: fn(int) -> string, cfg: RunConfig) -> PropertyReport`
- `forall_int_pair(...) -> PropertyReport`
- `forall_int_pair_with_config(...) -> PropertyReport`
- `forall_int_array(...) -> PropertyReport`
- `forall_int_array_with_config(...) -> PropertyReport`

### Reports

- `report_passed(report: PropertyReport) -> bool`
- `report_summary(report: PropertyReport) -> string`

The report fields include:
- `passed: bool`
- `case_count: int` (how many passing cases before first fail, or total passes)
- `discard_count: int`
- `shrink_count: int`
- `counterexample: string` (includes minimal failing input + message when failing)

## Reproducibility (seeded determinism)

Property runs are deterministic given a `seed` in `RunConfig`.

Recommended defaults:
- **CI**: fixed seed (e.g. `1`) for determinism
- **Local debugging**: pick a seed, lower trials, iterate quickly

Example with explicit config:

```nano
from "modules/proptest/proptest.nano" import int_range, forall_int_with_config, config, report_passed, report_summary, prop_pass, prop_fail

fn prop_no_zero(x: int) -> string {
    if (== x 0) { return (prop_fail "zero") } else { return (prop_pass) }
}

shadow prop_no_zero {
    let gen: proptest.IntRangeGenerator = (int_range -5 5)
    let cfg: proptest.RunConfig = (config 50 40 200 1234)
    let report: proptest.PropertyReport = (forall_int_with_config "no_zero" gen prop_no_zero cfg)
    assert (not (report_passed report))
    (println (report_summary report))
}
```

## Shrinking (how to interpret counterexamples)

When the first failure happens, the runner attempts to **shrink** the failing input:
- it searches for a simpler failing value (bounded by `max_shrink_steps`)
- the final minimal failing input is included in `report.counterexample`
- `report.shrink_count` tells you how much shrinking was done

Operationally:
- if you see a counterexample like `... x=0 :: zero`, focus on the property at that input
- if you need a different shrinking strategy, change the generator (or add a new shrinker in `modules/proptest/proptest.nano`)

## Writing good properties

- **Pure functions**: avoid I/O and global state inside properties.
- **Small, focused properties**: one invariant per property.
- **Use `prop_discard` for preconditions**: it keeps failure signals meaningful.
- **Short failure messages**: they end up embedded in `counterexample`.

## CI + local workflow

Property tests are normal `.nano` tests under `tests/` that compile + run like any other test.

- **Run locally**:

```bash
make test
```

- **CI pattern**:
  - keep trials modest (fast)
  - use fixed seeds (reproducible)
  - assert `report_passed(report)` in `shadow` tests

