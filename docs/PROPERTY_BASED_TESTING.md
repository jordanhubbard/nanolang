# Property-Based Testing (proptest)

I include a lightweight, deterministic property-based testing helper module: `modules/proptest/proptest.nano`.

This is the single canonical document for:
- Writing properties and generators
- Understanding how I shrink counterexamples
- Reproducing failures deterministically with seeds
- Running tests in CI

## Mental model

A property test is code that:
- generates inputs
- checks a property
- shrinks to a simpler counterexample on failure
- reports a `PropertyReport` you can assert on in a `shadow` test

I don't use special syntax. My properties are plain functions that return encoded outcomes.

## Outcomes (pass / fail / discard)

Your property function returns an encoded `string` outcome:
- `(proptest.prop_pass)` - the case passes
- `(proptest.prop_fail "message")` - the case fails and I start shrinking
- `(proptest.prop_discard "reason")` - the precondition is not met and I try another case

I use discard for preconditions like division by zero. I don't use it for random retries.

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
- `case_count: int` (passing cases before first fail, or total passes)
- `discard_count: int`
- `shrink_count: int`
- `counterexample: string` (minimal failing input and message when failing)

## Reproducibility (seeded determinism)

My property runs are deterministic when you provide a `seed` in `RunConfig`.

My recommended defaults:
- **CI**: use a fixed seed for determinism
- **Local debugging**: pick a seed, lower trials, and iterate

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

When I encounter the first failure, I attempt to shrink the failing input:
- I search for a simpler failing value, bounded by `max_shrink_steps`
- I include the final minimal failing input in `report.counterexample`
- `report.shrink_count` tells you how many shrinking steps I took

How to use this:
- if you see a counterexample like `... x=0 :: zero`, focus on the property at that input
- if you need a different shrinking strategy, change the generator or add a new shrinker to my proptest module

## Writing good properties

- **Pure functions**: avoid I/O and global state inside properties
- **Small, focused properties**: one invariant per property
- **Use `prop_discard` for preconditions**: it keeps failure signals meaningful
- **Short failure messages**: I embed these in the counterexample

## CI + local workflow

Property tests are normal tests under `tests/` that compile and run like any other test.

- **Run locally**:

```bash
make test
```

- **CI pattern**:
  - keep trials modest
  - use fixed seeds
  - assert `report_passed(report)` in `shadow` tests
