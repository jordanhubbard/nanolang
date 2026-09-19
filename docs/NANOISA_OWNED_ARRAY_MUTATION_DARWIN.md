# I qualify my unchanged mutation runtime on Darwin

I track task_d5da9c17547f495793d4359c57cadbee separately from completed Linux
runtime task5652 and retain parent430220/source mutation obligations. My baseline
is actual847 merge a52d990d7a0883ec5692be0f03872e0aaa1d54d9. I change no runtime,
producer, fixture assertions or admission. I seek review before setup builds or
new fixture execution.

## I isolate the host and compiler identities

I use puck.local (reported hostname puck, arm64), in a new checkout at
/private/tmp/nanolang-owner-array-mutation-d5da. I retain evidence in a separate
/private/tmp/nanolang-owner-array-mutation-d5da-evidence directory. I do not
modify the peer's CXWWHGGJX0 source checkout, caches or qualified tools.
Read-only discovery observes Apple clang17.0.0 (clang-1700.6.3.2), Homebrew
clang23.1.1 and SDK26.2. I pin actual versions, resolved executable paths/hashes,
SDK path/version, host OS and compiler configuration again at setup. The short
SSH alias puck did not resolve; puck.local succeeded. No fixture ran during
these host queries.

I select /usr/bin/clang explicitly for ordinary providers and ordinary fixtures;
I inventory its xcrun-resolved Xcode executable as well as the launcher. I select
/opt/homebrew/opt/llvm/bin/clang explicitly for sanitizer fixtures and inventory
its resolved Cellar executable, configuration file and sanitizer runtime libraries.
I use /opt/homebrew/bin/python3 and /usr/bin/make (GNU3.81), inventorying actual
resolved files. SDKROOT comes from xcrun --show-sdk-path. No compiler installation
or shared cache change is part of this task.

## I prepare once, then execute direct fixture recipes

I clone/fetch the canonical pin into the isolated tree and verify all twelve
Linux-sealed inputs against their qualified identity, with only canonical Make
integration differences separately explained. Before the build I archive source
and compiler identities. A temporary supplemental Makefile includes the canonical
Makefile.gnu and defines only a preparation target depending on
$(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS).
I invoke it with explicit CC=/usr/bin/clang and bounded parallelism. It executes
no test target. I retain the exact setup command, generated supplemental file,
first terminal and provider inventory. Any setup correction is reviewed and
recorded before a fresh attempt.

After preparation I use make -n for test-owned-array-mutation-runtime and
test-private-owned-array-runtime only to extract their exact published object
lists and linker flags. I verify every listed provider exists. I archive those
providers, compiler executables/configuration and exact recipes, then invoke
unittest directly; no phase rebuilds ordinary providers. Every phase receives its
own new artifact directory and before/after provider/tool/source inventory.
I stop on any drift or first failed phase. Inventories do not claim to cover
every transitive system tool or SDK file.

| Phase | Explicit compiler | Flags and purpose |
| --- | --- | --- |
| Apple ordinary mutation | /usr/bin/clang | Existing strict fixture flags; both dispatches/fusion modes/four APIs and native O0/O2 |
| Homebrew sanitizer mutation | /opt/homebrew/opt/llvm/bin/clang | -fsanitize=address,undefined -fno-omit-frame-pointer; same complete fixture |
| Apple ordinary adjacency | /usr/bin/clang | Unchanged private runtime corpus validates configurable-main reuse |

The sanitizer phase sets ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 and
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1. I use actual Homebrew sanitizer
support; I do not silently disable leak detection or substitute Apple coverage.
Instrumented fixture/VM/heap/emitter/generated native translation units remain
distinct from ordinary linked providers. I retain explicit macro evidence for
true switch and computed-goto, exact emitted C equality, allocation site records,
1906 mutation checks per VM binary, cleanup and output assertions unchanged.
I preserve actual observed counts rather than infer passes from an exit alone.

I bound setup and each gate with a Python subprocess deadline of600 seconds and
terminate its process group on expiry, recording the timeout distinctly from a
program assertion. The complete gate driver has an outer1800-second bound.
No historical failed artifact is executed. First outcomes, all logs and artifacts
are retained. After review I seal exact hashes and report only this Darwin runtime
supplement; paired source mutation and parent430220 need their own acceptance.
