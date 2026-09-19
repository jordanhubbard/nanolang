# I freeze owner-array source qualification after runtime acceptance

This is a preparation recipe for task_18731b55c66846f9826290148c967ca5.
It is not an execution report. My reviewed producer checkpoint is99fdb4804 and
my approved six-method fixture checkpoint isaae7e8883. Public activation is still
qualifying; I will record its actual qualified commit rather than substitute an
API checkpoint or assume a passing private runtime implies public acceptance.

## I integrate without changing retained trees

I create a new branch/worktree from this source branch only after the runtime
agent supplies the qualified activation commit. I merge the actual canonical
activation ancestry there. I retain the current source and runtime qualification
trees and their tools unchanged. I inspect every conflict before building and
send any altered production resolution for review. Expected source production
must compare byte-for-byte with99fdb4804; expected activation providers must
compare with the qualified runtime pin. Additive documentation is separate.

My static closure inventory includes these paths and their transitive includes:

- `src/nanoisa/owned_array_admission.h`, `owned_array_admit.inc`, and its inclusion
  from `verifier.c`; route/status spellings and checked parameter-view correction.
- `owned_array_authority.h`, its implementation in the affine providers, retained
  layout/origin includes, and both public and private query boundaries.
- VM dispatch/entry providers, converter, nvm2c selection and owned native emitter;
  service rejection must precede owner-array selection, which precedes Samples.
- `src/nanovirt/wrapper_gen.c` object closure and `modules/nanoisa/module.json`
  source closure, including `mixed_float_proof.o`/`.c`. An included `.inc` is not
  a separately linked object. Generated standalone runtime embedding/check targets
  remain prerequisites supplied by the integrated Makefile.
- `GNUmakefile` includes `Makefile.gnu`; `Makefile` is only the BSD bridge.
  I use the actual GNU recipes, never infer targets from binary basenames.

Before starting, I capture the exact merged HEAD, clean tracked status, resolved
host compiler/linker/archive/Python/GNU-make hashes, SDK and environment selectors,
all tracked build/fixture inputs (including GNUmakefile and Makefile.gnu), and
existing bin/obj/lib files. No copied sentinel, compiler, module cache or old
object from a qualified tree is a fresh bootstrap input.

## I build the actual prerequisites in bounded phases

My inspected recipes provide `bootstrap`, `nanoisa_emit`, `nano_virt`, `nano_vm`,
`nvm2c`, `nanoisa_dump`, and `test-local-binding-metadata`. The last target also
runs its declared allocation/marker controls and leaves `obj/test_local_bindings`.
`nanoisa_dump` produces `bin/nanoisa`; there is no inferred `nanoisa` build target.

I use `make` on Linux and the resolved GNU `gmake` on Darwin. My first phase is
`-j2 -f Makefile.gnu bootstrap`, with a retained log/status and a1800-second outer
bound. I stop on its first nonzero/timeout result. A second phase uses the same
Makefile plus a retained external setup fragment:

```make
.PHONY: owned-array-source-setup
owned-array-source-setup: nanoisa_emit nano_virt nano_vm nvm2c nanoisa_dump test-local-binding-metadata
	$(CC) $(CFLAGS) -o obj/borrow_shadow_names tests/nanovirt/borrow_shadow_names.c $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
```

The setup invocation names `owned-array-source-setup` explicitly and retains its
own1800-second bound/status. It must leave nanoc_c, nanoc_stage1, nanoc_stage2,
nano_virt, nano_vm, nanoisa, nanoisa_emit, nvm2c, the two probes and wrapper runtime
objects available. I record the exact actual inputs and outputs, not merely a
successful command exit.

I remove inherited NANOC/tool/cache overrides and explicitly set NANO_VM,
NANO_NVM2C, NANO_AOT_RUNTIME, NANO_MODULE_PATH and NANO_BUILD_CACHE to this new
tree's bin/modules/obj paths, plus PYTHONPATH to its root. I retain these values
in the invocation report. No external compiler cache is reused.

## I separate producer setup from source assertions

The existing SourceBorrowEmission setup produces Stage1/Stage2 raw emitters and
three shadow drivers. Its five compiler invocations keep the existing900-second
setup bounds; ordinary source commands keep180seconds. I retain their paths,
outputs, actual compiler commands and hashes. I capture a new post-setup snapshot
of tools, libraries, objects and module-cache files before the six test methods.
A cache generated during setup is an output of that phase, not evidence that
bootstrap inputs stayed unchanged through setup.

The bounded runner loads only `tests.test_source_owned_float_arrays` through its
explicit `load_tests`; inherited tests are not independently discovered. It
wraps that class's original setup to capture the post-setup snapshot and select
the native compiler before returning to unittest. It preserves first-failure
status and retained fixtures. It does not run a separate setup and then let
unittest silently run setup a second time.

On Linux I pin the native `CC` executable explicitly (initial GCC qualification).
On Darwin I pin producer build selection and `xcrun --sdk macosx --show-sdk-path`
separately, then set `CC=/opt/homebrew/opt/llvm/bin/clang` after producer setup.
I capture the resolved Homebrew compiler hash/version, SDK and dynamic-library
inputs. The inherited execute_pair reads CC directly, invokes strict C11/O2 with
ASan+UBSan and `-lm`, and runs with `detect_leaks=1:halt_on_error=1`.
NANO_NATIVE_TEST_CC does not control that method. I do not assume make's command
line CC propagates to Python subprocesses. Source fixtures do not invoke a
separate Clang/Wasm selector; any later neighboring gate gets its own audited
selector inventory before execution.

## I retain evidence and keep acceptance bounded

Each phase has one terminal status/log, source/tool maps before and after, exact
commands and durations. I hash emitted assemblies/modules/native binaries and
all six retained producer/driver inputs before packaging. Existing snapshot keys
must remain equal; legitimate newly generated objects are separately listed and
hashed rather than making a false whole-map equality claim. Packaging and commit
steps run sequentially with checked subprocess results; no failed assertion can
publish a partial successful manifest.

I stop at the first demonstrated failure and record it before repair. No rejected
module executes. I keep source, VM/native sanitizer and platform results distinct.
Mutation set/push/length, full430220/4be and product/release gates remain open.
