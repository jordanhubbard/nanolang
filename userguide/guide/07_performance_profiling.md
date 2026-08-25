# Performance Profiling

I wrap a native binary compiled with `-pg` so that running it collects a host
profile and prints JSON an LLM can read. I do not guarantee a speedup. I keep
an optimization when a second profile and my tests show it helped.

The canonical field list, re-entry environment variables, and `/tmp` artifact
names are in [docs/PERFORMANCE_MONITORING.md](../../docs/PERFORMANCE_MONITORING.md).
Older links to [Chapter 8](../08_profiling.md) reach a pointer to this page.

## Capture a profile

```bash
./bin/nanoc program.nano -o bin/program -pg --profile-output profile.json
./bin/program
```

The first process is a coordinator (`_nl_run_with_profiling`). It creates a
uniquely named artifact from its process ID, launches the measured workload in
a child, waits, converts the collector output to JSON, and deletes the
temporary files. Concurrent runs do not share the same `/tmp` path.

JSON is written to `profile.json` and also to **stdout** (between two banner
lines). Redirecting stderr will not capture it.

## Flags that are not `-pg`

| Flag | Role |
| --- | --- |
| `-pg` / `--profile-output` | OS collector + JSON. Native `main` becomes `_nl_run_with_profiling`. |
| `--profile` | Timing hooks in generated C. Table on stderr. |
| `--profile-runtime` | Also write `.nano.prof` collapsed stacks. Native backend only. |
| `--pgo <file>` | Inline using a `.nano.prof`. Not using `-pg` JSON. |

C flags added with `-pg`: `-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls`.

## Platform boundary

| Platform | Collector | Measurement |
| --- | --- | --- |
| macOS with full Xcode | `xctrace` Time Profiler | sampling a launched child |
| macOS fallback | `sample` | periodic sampling of a synchronized child |
| Linux | `gprofng collect app` | instrumented collection in a child process |
| Other OS | none | I print that profiling is not supported and run the program |

I treat xctrace as available only when `which xctrace` and `xctrace version`
both succeed. Command Line Tools alone is not enough.

On macOS, the `sample` fallback creates the workload child first and holds it
on a pipe (`_NL_PROFILING_ACTIVE`, `_NL_PROFILING_CWD`). A sampler child
attaches to that PID (`sample <pid> 60 -f /tmp/nanolang_sample_<pid>.txt -mayDie`),
then the coordinator releases the workload. Short runs can still finish before
`sample` attaches.

On Linux, re-entry uses `_NL_PROFILING_CHILD` or `LD_PRELOAD` containing
`libgp-collector`. If `gprofng` is missing, the child runs without a collector.
gprofng is instrumented collection. The JSON field `profile_type` is still the
string `"sampling"` on every OS; that is a historical label, not a claim that
Linux samples.

## JSON I actually emit

Each hotspot has only `function`, `samples`, and `pct_time`. I do not emit
source locations or per-call microseconds.

```json
{
  "profile_type": "sampling",
  "platform": "macOS",
  "tool": "xctrace",
  "binary": "./bin/program",
  "hotspots": [
    {"function": "nl_hot_function", "samples": 120, "pct_time": 12.0}
  ],
  "analysis_hints": [
    "Functions with high sample counts are hot spots",
    "Look for nl_ prefixed functions (NanoLang generated)",
    "str_ and array_ functions often indicate algorithmic issues",
    "Deep call stacks may indicate recursion or callback chains"
  ]
}
```

`tool` is `"gprofng"`, `"xctrace"`, or `"sample"`. On Linux, `samples` is
exclusive percent times ten, not a true sample count. On xctrace I emit the
top 20 names that start with `nl_`.

Treat a profile as measurement of one workload on one machine.

## Tune with an LLM

1. Compile with `-pg --profile-output`.
2. Run a representative workload, not a one-line shadow.
3. Give an LLM the JSON, the relevant source, and the command that produced
   the workload. Ask for one change tied to a measured hotspot.
4. Run tests, then re-profile the same workload.
5. Keep the change only if tests still pass and the numbers improve.

An LLM can suggest an optimization; the profile can test its effect. Neither
substitutes for the other. Compare wall-clock **without** `-pg` when you report
speed.

`--pgo` reads `.nano.prof` from `--profile-runtime`. That file is not a PGO
input from `-pg` JSON.
