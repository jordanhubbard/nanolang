# I supply the host runtime required by my standard-library artifact

My generated compiler previously aborted in `dlopen` for `path_canonical`.
LLDB and `dlerror()` identified the unresolved `_dyn_array_get_string` symbol.
The library also requires allocation, string-push and GC entry points. Scalar
calls do not avoid those dependencies under `RTLD_NOW`.

I added `make nvm2c-runtime`, which combines my existing array, GC and GC-struct
objects into `bin/nano_aot_runtime.o`. I retain all object symbols, including
those referenced only by dynamically loaded modules. I link this object in my
native compiler acceptance test. My CLI help points to
[`docs/AOT_RUNTIME.md`](../AOT_RUNTIME.md), which documents macOS and Linux
commands and Linux's required dynamic export flag.

I do not duplicate the runtime implementation, add an independent module GC,
replace artifact calls with builtins, or remove adapter validation.

## I checked the actual boundary

`test_real_std_artifact_uses_host_runtime` builds my real `fs.c` and `process.c`
as a shared library without its own runtime. Its NanoISA module calls
`path_canonical` and `fs_walkdir` through exact artifact imports. Without the
host runtime the executable traps. Linked with the host object, it walks one
file, checks the copied filename length after foreign result release, and
verifies that the host GC object count returns to its original value.
This checks foreign allocation/release balance, not bounded AOT memory use.

On this macOS host:

- The real-library regression passes under strict C11 compilation.
- `make test-refcount-gc test-gc-struct` passes.
- `make test-one-ir-compiler` passes 16 of 17 methods. Native compiler emission,
  C compilation and `--help` pass; compiling hello still aborts.
- `git diff --check` passes.

I traced the remaining abort with the runtime linked. `nl_parser_is_at_end`
expects record field zero's storage kind to be 2, but the constructed field has
kind 0 and integer payload 24. I retain the failing check and acceptance test.
MAC `task_45fedd409e1447089dad970396b0a075` tracks that distinct failure.

The broader `test-dyn-array` invocation also runs bootstrap; its stage-two
compilation was still running at this checkpoint. I do not report that gate as
passed. Linux link flags are documented but were not executed on this host.

MAC `task_fca5000374ef49988da33e5617e703b8` tracks this linkage fix. Worker claims
still fail with `agent_status_unavailable`; I record evidence without forcing
ledger completion. This checkpoint is not a release.
