# Explicit artifact string cleanup

I preserve the legacy non-NULL filesystem reader while allowing its provider
to release original result storage after a consumer snapshots it. My
[contract](../NATIVE_ARTIFACT_STRING_LIFETIMES.md) defines the exact versioned
companion, same-image check, bounded direct-call signature and absent-companion
compatibility. Source checkpoint `c64323b1` introduces the implementation;
`88a1c09c` includes the optional generated-helper reference and tests.
Integrated code pin `a349b3b0a0e5bfbf26ffefa253d447c4e122d672` includes merged
scalar comparisons and caller-reference support.

At the integrated pin, twelve focused methods passed in 7.013s with GCC and
7.492s with Clang. Eight new artifact methods cover zero/one/two arguments,
repeated allocation/release counts and aliases, borrowed no-companion results,
two libraries exporting the same function, wrong-image companion refusal,
NULL result cleanup, real file contents and empty/missing/NUL cases, provider
allocation failure, and consumer copy failure in both VM and native C. The
other four methods retain existing builtin and facade string coverage.
Generated native programs use ASan/UBSan/LSan; the VM copy-failure test compiles
an isolated bridge shim and verifies provider allocation/release counts. I do
not describe the ordinary VM CLI as sanitizer-instrumented.

Before the additive main integration, full native and shape gates passed
2,414 and 1,092 checks, and all 28 existing VM FFI cases passed with their
array ABI, callback and retained-image prerequisites. After integration,
1,548 caller-reference checks, 43 owner-allocation and 55 parameter-allocation
checks passed; all 28 VM FFI cases and their prerequisites passed again.

Commands include:

```sh
make nanoisa_dump nano_vm nvm2c
python3 -m unittest -v tests.test_artifact_string_release tests.test_native_host_strings
make test-nvm2c test-vm-ffi
make test-caller-references test-vm-ffi
```

For the Clang run I selected the installed GCC 13 support directory explicitly
through a private cc wrapper; I did not suppress compiler warnings.

I retain setup and interrupted-run evidence separately. The initial existing
string test invocation omitted the nanoisa_dump build prerequisite and could
not find bin/nanoisa. The first eight-method test run had an incorrect expected
VM diagnostic; that assertion was corrected before the final combined runs.
During the first Clang run, I merged main while its isolated VM bridge check
was active; that run failed and is not acceptance evidence. I retained its
log, rebuilt the complete combined source, held it unchanged and ran both
compiler variants successfully. Logs remain under
`/tmp/nanolang-artifact-release-*`.

After PR #582, I resolved only appended Make targets, retaining both. The eight
artifact methods passed again in 7.838s at integrated code `2a8627ff`. The first
local resolution script stopped before writing; make rejected the retained
markers before tests. I corrected the local merge and retained that setup log.

This completes the bounded file_read adapter contract in MAC
`task_bdc323f270d44f02b38ba728f1f93184`. Broader host-result ownership, other
filesystem helper contracts and callback/co-process/interpreter artifact
enrollment remain outside this repair. Product PR #522 and release publication
remain held; these tests do not establish a cause for its compiler abort.
