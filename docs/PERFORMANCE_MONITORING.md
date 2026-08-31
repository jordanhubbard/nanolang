# Performance Monitoring and LLM Optimization

I compile to C. When you pass `-pg`, I also wrap the generated `main` so that
running the binary launches the host profiler's collector, then I turn that
host output into JSON an agent can read.

This is measurement, not a promise of speed. I keep a change only when a
re-profile (and my tests) show it helped.

The canonical field list and OS wrapper details are in
[docs/PERFORMANCE_MONITORING.md](PERFORMANCE_MONITORING.md).
The published user-guide session is
[userguide/guide/07_performance_profiling.md](../userguide/guide/07_performance_profiling.md).
A measured walk-through lives in [PROFILING_CASE_STUDY_PRIMES.md](PROFILING_CASE_STUDY_PRIMES.md).
Characteristics of generated C (allocation, loops) live in [PERFORMANCE.md](PERFORMANCE.md).

## Three tools that are not the same thing

| Flag | What I do | Output | Use it for |
| --- | --- | --- | --- |
| `-pg` plus optional `--profile-output <path>` | Wrap native `main` with `_nl_run_with_profiling`. Compile the C with `-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls`. | JSON (`profile_type` is always `"sampling"`) on **stdout**, and the same JSON body in `<path>` when `--profile-output` is set | Agent-readable hotspots from the OS sampler |
| `--profile` | Instrument generated C with timing hooks | Text table on **stderr** at exit; `NANO_PROFILE=0` disables collection without rebuilding | Call counts and time in the native backend |
| `--profile-runtime` / `--profile-runtime-output <p>` | Implies `--profile`. Native backend only | Collapsed stacks in `.nano.prof` (default `<binary>.nano.prof`) | Flame graphs; this file is what `--pgo` reads |
| `--pgo <file>` | Profile-guided **inlining** from a `.nano.prof` | A faster native binary if the profile matches the call sites | Not driven by `-pg` JSON |

`--profile-runtime` is rejected on non-native backends. I emit that error in
`src/main.c`; I do not silently drop the flag.

## Runtime Toggle for Generated-C Profiling

I read `NANO_PROFILE` once when a generated executable starts. A binary built
with `--profile` collects timing data by default. Set `NANO_PROFILE=0` to
disable the timing and flamegraph hooks without recompiling; set
`NANO_PROFILE=1` to enable them explicitly. The generated function hook checks
only its cached boolean, so disabled profiling does not perform environment
lookups or timing calls.

`-pg` JSON is not a PGO input. `--pgo` reads `.nano.prof`.

## Compile and capture JSON

```bash
./bin/nanoc program.nano -o bin/program -pg --profile-output profile.json
./bin/program
# profile.json is the JSON object. stdout also prints banners plus the same object.
```

Do not redirect stderr and expect JSON. I print the JSON with `_nl_profile_emit`,
which writes **stdout** (and the `--profile-output` file). The banners
`PROFILE ANALYSIS (LLM-READY JSON)` and `END PROFILE ANALYSIS` are stdout too.

Batch Linux collection of the compiler's own tests is `scripts/profile_tests.sh`.

## JSON I actually emit

Generated in `generate_profiling_system` (`src/stdlib_runtime.c`). Each hotspot
is only `function`, `samples`, and `pct_time`. I do not emit `location`,
`calls`, or `per_call_us` in this JSON.

```json
{
  "profile_type": "sampling",
  "platform": "Linux",
  "tool": "gprofng",
  "binary": "./bin/program",
  "hotspots": [
    {"function": "nl_process_pixels", "samples": 684, "pct_time": 68.4}
  ],
  "analysis_hints": [
    "Functions with high sample counts are hot spots",
    "Look for nl_ prefixed functions (NanoLang generated)",
    "str_ and array_ functions often indicate algorithmic issues",
    "Deep call stacks may indicate recursion or callback chains"
  ]
}
```

`profile_type` is the string `"sampling"` on every OS, including Linux gprofng.

`tool` is `"gprofng"`, `"xctrace"`, or `"sample"`.

`samples` is a count derived from the host tool, not a portable unit:

- **Linux / gprofng:** exclusive percent times ten, truncated to `int`
  (`(int)(excl_pct * 10)`). It is a stand-in so the field is populated.
- **macOS / xctrace:** how many `name="..."` attributes matched that symbol
  while parsing the exported Time Profiler table.
- **macOS / sample:** the sample command's stack counts for lines at or above
  1% of the reported total.

I emit at most 20 hotspots. On xctrace I keep only names that start with `nl_`.
On gprofng and `sample` I keep rows the parser accepts (gprofng: exclusive
percent ≥ 1%; `sample`: percent ≥ 1%).

## What `-pg` does at runtime

The generated program's `main` becomes `_nl_run_with_profiling`. That wrapper
forks a collector, waits, parses, prints JSON, then deletes temp files under
`/tmp`.

### Linux

1. If `_NL_PROFILING_CHILD` is set, or `LD_PRELOAD` contains `libgp-collector`,
   I call `real_main` and return. That is the re-entry path when gprofng
   launches me again.
2. Otherwise I `fork`. The child sets `_NL_PROFILING_CHILD=1` and
   `execvp`s `gprofng collect app -o /tmp/nanolang_gprofng_<pid>.er` followed
   by the original `argv`.
3. If `gprofng` is missing, `execvp` fails and the child `_exit((int)real_main())`.
   The parent still runs `gprofng display text -functions` on the experiment
   directory; that display may be empty.
4. The parent waits, parses `gprofng display text -functions`, prints JSON,
   then `rm -rf` the `.er` directory.

gprofng comes from binutils. I do not ship it.

### macOS

I treat xctrace as available only when `which xctrace` succeeds **and**
`xctrace version` succeeds. Command Line Tools alone is not enough; I need
full Xcode for `xctrace version`.

**xctrace path**

1. If `_NL_PROFILING_ACTIVE` is set, restore `_NL_PROFILING_CWD` with `chdir`
   and run `real_main`. `xctrace record --launch` re-executes the binary; this
   guard stops a second nested collector.
2. I set `_NL_PROFILING_CWD` and `_NL_PROFILING_ACTIVE=1`, then fork
   `xctrace record --template Time Profiler --output /tmp/nanolang_trace_<pid>.trace --launch --` plus the original `argv`.
3. After xctrace exits I export the Time Profiler table to
   `/tmp/nanolang_table_<pid>.txt`, parse `name="` frames, emit the top 20
   `nl_`-prefixed names, then `rm -rf` the `.trace` bundle and unlink the table.

Time Profiler samples. The wrapper may print "instrumentation" on stderr; that
string is not a second profiler mode.

**`sample` fallback** (when xctrace is missing or `xctrace version` fails)

1. Set the same re-entry env vars.
2. Fork the program; the child blocks on a pipe until the parent is ready.
3. Fork `sample <child-pid> 60 -f /tmp/nanolang_sample_<pid>.txt -mayDie`.
4. Sleep 200 ms, signal the child, wait, then parse the call graph.
5. Unlink the sample file.

Short runs can finish before `sample` attaches. I warn on stderr if the
sample file cannot be read.

### Other operating systems

I print `[profile] Profiling not supported on this platform` on stderr and run
`real_main`. The `-pg` C flags still apply to the compile.

## LLM loop I will actually run

1. Compile with `-pg` and `--profile-output`.
2. Run a **representative** workload (the profile of a tiny test is not the
   profile of the program).
3. Read the JSON. Treat `nl_` names as generated functions from my source.
4. Propose a source change aimed at a hotspot.
5. Run **tests** (shadows, then the suite that covers that code).
6. Re-profile the same workload.
7. Keep the change only if the new profile (or a wall-clock measurement
   without `-pg`) shows an improvement. Discard it if tests fail or the
   hotspot did not move.

I do not claim a speedup factor. One sieve rewrite measured 5.6× in
[PROFILING_CASE_STUDY_PRIMES.md](PROFILING_CASE_STUDY_PRIMES.md); that is a
single measured case, not a bound.

Overhead: collectors slow the process. Compare optimized binaries **without**
`-pg` when you report wall-clock results.

## Native instrumentation (`--profile`)

`--profile` injects timing in generated C (`generate_instrumented_profiling_system`).
The report is a stderr table: function, calls, total ms, average µs.

`--profile-runtime` writes flamegraph collapsed stacks (`fn_name count`) for
`flamegraph.pl`. `--pgo` uses that file to decide inlines. It does not read
`-pg` JSON.

## Implementation map

| Piece | Where |
| --- | --- |
| `-pg` / `--profile-output` flags | `src/main.c` |
| C compile extras | `profile_flags` in `src/main.c` |
| Emit wrapper | `generate_profiling_system` in `src/stdlib_runtime.c` |
| Call wrapper instead of `main` | `src/transpiler.c` when `env->profile_gprof` |
| `--profile` / flamegraph | `generate_instrumented_profiling_system`, `generate_flamegraph_profiling_system` |
| `--pgo` | `src/pgo_pass.c` |
