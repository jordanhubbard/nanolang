# Performance Profiling

I can compile a native program with a profiling wrapper built into the binary. Use `-pg` to enable it and `--profile-output` to give the structured result a stable path:

```bash
bin/nanoc -pg --profile-output profile.json program.nano -o program
./program
```

The first process is a coordinator. It creates a uniquely named profile artifact from its process ID, launches the measured workload in a child process, waits for it, converts the profiler's result to JSON, and removes the temporary artifact. This keeps concurrent profiling runs from writing to the same collector data.

## Platform Boundary

The child-process arrangement is shared, but the measurement method is platform-specific:

| Platform | Collector | Measurement |
| --- | --- | --- |
| macOS with full Xcode | `xctrace` Time Profiler | sampling through a launched child |
| macOS fallback | `sample` | periodic sampling of a synchronized child |
| Linux | `gprofng collect app` | instrumented collection in a child process |

On macOS, the `sample` fallback creates the workload child first and holds it on a pipe. A sampler child attaches to that exact PID, then the coordinator releases the workload. The output file also contains the coordinator PID. This is the unique child-process sampling mechanism: the sampler measures the intended run, including short-lived programs, without colliding with another profile in `/tmp`.

On Linux, I still isolate the measured run in a uniquely named child experiment, but `gprofng` performs instrumented collection rather than the macOS `sample` mechanism. I do not call both methods sampling merely because their JSON has a historical `profile_type` value of `sampling`.

The generated native binary also receives these compiler flags:

```text
-pg -g -fno-omit-frame-pointer -fno-optimize-sibling-calls
```

`-pg` profiling is available on the native C path. The narrower direct backends do not use this wrapper.

## Read The Result

The JSON reports the platform, collector, binary, hotspots, sample-like counts, and percentage of measured time. Generated NanoLang functions normally have an `nl_` prefix.

```json
{
  "profile_type": "sampling",
  "platform": "macOS",
  "tool": "sample",
  "binary": "./program",
  "hotspots": [
    {"function": "nl_render_frame", "samples": 842, "pct_time": 63.4}
  ]
}
```

Treat a profile as measurement of one workload on one machine. It is not a proof that a function is intrinsically slow.

## Tune With An LLM

Give an LLM the profile, the relevant source, and the command that produced the workload. Ask for a small change tied to a measured hotspot:

```text
This profile came from `./program fixtures/large.scene`.
Identify the highest-cost NanoLang function, explain the likely cause from the
source, and propose one minimal optimization. Preserve behavior and shadows.
Do not optimize functions unsupported by the profile.
```

Then verify behavior and measure again:

```bash
make test
bin/nanoc -pg --profile-output profile-after.json program.nano -o program
./program fixtures/large.scene
```

Compare repeated runs under the same workload. Keep a change only when tests still pass and the measurements improve consistently. An LLM can suggest an optimization; the profile can test its effect. Neither substitutes for the other.

For compiler-directed profile-guided inlining, use `--profile-runtime` to produce a `.nano.prof` file and pass it back with `--pgo`. That is a separate mechanism from the `-pg` child-process profiler described here.
