# Recovered-work integration benchmark (2026-09-02)

Measured with `scripts/benchmark_nanoisa.sh`, 8 samples per workload.

Three points on the same machine and toolchain:

- **control** — `main` plus the schema-test fix (#180)
- **+6** — control plus the first six recovered branches
- **+9** — all nine recovered branches

## Wall time (median, ms)

| workload | control | +6 | +9 | control→+6 | +6→+9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nl_array_complete` | 19.5 | 20.0 | 27.1 | +2.5% | +35.7% |
| `nl_extern_math` | 19.5 | 19.9 | 20.5 | +2.1% | +3.1% |
| `nl_fibonacci` | 20.0 | 20.2 | 22.1 | +1.2% | +9.2% |
| `nl_forth_interpreter` | 20.8 | 21.4 | 23.1 | +3.0% | +7.8% |
| `nl_function_variables` | 19.5 | 20.0 | 21.9 | +2.7% | +9.3% |
| `nl_hashmap_word_count` | 19.4 | 19.9 | 21.8 | +2.5% | +9.6% |
| `nl_string_operations` | 19.5 | 19.8 | 22.1 | +1.4% | +11.8% |

## Heap allocation calls

| workload | control | +6 | +9 |
| --- | ---: | ---: | ---: |
| `nl_array_complete` | 35 | 35 | 44 |
| `nl_extern_math` | 22 | 22 | 37 |
| `nl_fibonacci` | 107 | 107 | 82 |
| `nl_forth_interpreter` | 16 | 16 | 388 |
| `nl_function_variables` | 41 | 41 | 44 |
| `nl_hashmap_word_count` | 4 | 4 | 7 |
| `nl_string_operations` | 5 | 5 | 20 |

## Retired instructions (unchanged)

| workload | control | +9 |
| --- | ---: | ---: |
| `nl_array_complete` | 244 | 244 |
| `nl_extern_math` | 167 | 167 |
| `nl_fibonacci` | 32082 | 32082 |
| `nl_forth_interpreter` | 402 | 402 |
| `nl_function_variables` | 193 | 193 |
| `nl_hashmap_word_count` | 109 | 109 |
| `nl_string_operations` | 78 | 78 |

## Reading

Retired-instruction counts are identical across all three points, so the same
work is being executed: the difference is cost per unit of work, not more work.

The first six recovered branches cost 1.2%-3.0% and leave allocation counts
untouched. Everything after costs 3.1%-35.7%, and allocation counts rise
sharply at the same point -- `nl_forth_interpreter` goes from 16 to 388
allocation calls.

That points at preinstantiated module constants (task_deba0dbe), which
allocates a VmString for every string in a module at init whether the program
touches it or not. The Forth interpreter references a small fraction of its
constant pool, so it pays for the whole thing.

This is a deliberate design property, not an oversight: the accompanying test
asserts that execution performs *zero* allocations, which is the roadmap item
read literally ("string literals do not allocate ... on every execution"). The
cost simply moved from execution to initialization, and on short-running
workloads like these the move is a net loss.

Making instantiation lazy (allocate on first PUSH_STR, cache thereafter) was
tried and reverted: it breaks that zero-allocation guarantee by design, and the
choice between "no allocation during execution" and "no allocation for unused
constants" is an architectural one, not a merge decision. Tracked separately.

These numbers are medians of 8 samples with p25/p75 recorded in the raw
summaries; no IQR overlap between control and +9 on any workload.


## Environment

```json
{
  "system": "Darwin",
  "release": "25.6.0",
  "machine": "arm64",
  "processor": "arm",
  "python": "3.14.7",
  "cc": "Apple clang version 21.0.0 (clang-2100.1.1.101)",
  "git_commit": "ad79647f89fe2ab5cc3b96d88b6f624e3073a75f"
}
```
