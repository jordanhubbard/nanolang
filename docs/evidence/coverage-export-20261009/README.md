# My effective linker flags

Coverage job113751347364 at a88c45c3b still fails the real emitter/native link with unresolved GCC coverage runtime symbols. The harness consumes LDFLAGS, but nested GNU Make drops its automatic environment export after my override additions. The retained Make probe reports LDFLAGS=None while MAKEFLAGS still carries the requested coverage option.

I explicitly export effective LDFLAGS. In Linux ARM64, my real component harness links a separately instrumented dyn_array/gc/gc_struct runtime through a nested copy of the actual Makefile. Before export it fails after 94.865 seconds with unresolved __gcov symbols; after export it passes after 116.564 seconds. I substitute only the runtime-object path and retain real compilation, VM execution, sanitizer execution and assertions. The module cache is private to this probe.

My permanent regression compiles an instrumented native object and requires a nested real Make recipe to link and execute it using only the propagated link flags. Hosted coverage remains required. I also retain bootstrap logs/manifests on CI failure so the newly reached Linux Stage 1 failure can be diagnosed; I do not assign it a cause from its exit status alone.
