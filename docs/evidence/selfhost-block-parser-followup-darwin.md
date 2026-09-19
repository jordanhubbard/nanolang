# I prepare the missing tools before Darwin leak checking stops the next gate

I record this separate corrected-setup attempt under
`task_bd8e2d91943ae17059d289e13ebc34fe` and discovered child
`task_11cf4518c79d5a832d9970982e551c3d`. The parser repair remains
`task_a613986ffa6f476293e3befa8d9accfd`. This evidence does not qualify an
installed product or authorize publication.

## Frozen checkout and copied compilers

I used a fresh detached checkout at exact
`603785c9c295cf2c780fea9633aa6d86f37744f4`:

- checkout: `/private/tmp/nanolang-parser-followup-gate.pZZgXm`
- evidence: `/private/tmp/nanolang-parser-followup-evidence.FS50W5`

The before, frozen and after maps each contain all6,431 tracked files and have
SHA-256
`51013841632ebcc2be76c602335a68a3586b86684294795364d5799cac37f3e9`.
The checkout remained clean.

I copied only the three compilers from the already qualified fresh-bootstrap
checkout. Their hashes match the prior evidence and remain unchanged after this
attempt:

| Compiler | SHA-256 |
|---|---|
| `bin/nanoc_c` | `7b1367443edf8066cdd414bcca7ccea57818b7cd1bbb6051602dbc6b53d5f2e9` |
| `bin/nanoc_stage1` | `86daed4b5bd8af59c535a6abee9a197f8481f8056422c599b67260555f017813` |
| `bin/nanoc_stage2` | `b00462b3b332bc6ab0d719b61ee3315e393b2b5eb52f177cb419698a747dca2b` |

I created `bin/nanoc` as a link to the copied Stage2 compiler. I did not run a
second bootstrap or repeat the already passing 36-source matrix or callable
methods.

## Corrected executable setup

The statically inventoried command passed in3.28 seconds:

```text
make -j8 bin/nano nano_virt nano_vm nanoisa_dump nvm2c
```

The log SHA-256 is
`7358ac71bad339edf2c647e71aec8097953be69b575368b7a0bdb4aedcb568ea`.
The freshly built executable hashes, unchanged after the gates, are:

| Executable | SHA-256 |
|---|---|
| `bin/nano` | `7c80be065b33e88a7edb1f1c58ed4bbf13402feecb6eaec00202320e35b92b2f` |
| `bin/nano_virt` | `7b478f586cee633ebd3498e890ffcddb5ef4bf3b1b19feee3e1c1dcff2991b98` |
| `bin/nano_vm` | `e292232dfe6785e7a39e3655e844d4eb56ca8e077e869cd456e9b2da5b168ffd` |
| `bin/nanoisa` | `c6971f00f2b6bdb4ce5561df09efb3db5804a4d61cb3a695fbb84a4313b6c310` |
| `bin/nvm2c` | `1c874c1d3b08ed89908f8d611463278e96366e61a6de6558d682e53f6314efb9` |

## Previously unreached union methods pass

I ran only the two union-result methods not completed by the preceding attempt.
Both passed in0.91 seconds. They cover four strict generated-C variants,
ordinary interpreter execution, NanoVirt emission, NanoVM verification and
NanoVM execution. The log SHA-256 is
`3980da33e84460c8f2e325cce261dcbf6fd4b66bd5a2d9a3c65bfb6d979fe509`.

Combined with the separately retained three callable passes and first union
method, this completes the bounded deeper public-C callable/union sequence. I
do not relabel those prior results as part of this checkout.

## First new terminal: Apple sanitizer runtime has no leak mode

Checked-owner selection built all three test-only drivers:

| Driver producer | Driver SHA-256 |
|---|---|
| C seed | `3b0461984a3fca2f8a47bdb6b8d20bcaa6accfee66dff54319cf8f1abedbb337` |
| Stage1 | `91b64c860362c4e6069486db5168054a2bdd8edcb46450abe951b568d0f4cb14` |
| Stage2 | `1433157eb2c8916de42c37e86efbb4ea31e2886abfb72569c4de7c4b745553cf` |

Its first `chain`/C-seed bytecode and VM route reached strict native execution.
The shared `execute_pair` helper compiled through unset `CC`, which selected
`/usr/bin/cc` and Apple Clang21.0.0, then ran with the unchanged
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`. The native process returned -6
with:

```text
AddressSanitizer: detect_leaks is not supported on this platform.
```

The fail-fast unittest command exited1 after202.91 seconds. Its log SHA-256 is
`f5f7d068ef1501be9517aed0b4505e289c8cd87dd8bcd657f249c3ba5840ea14`.
This is a host sanitizer-selection failure, not a NanoLang ownership result.

Static inventory after the terminal confirms that existing Homebrew
Clang23.1.1 is available at `/opt/homebrew/opt/llvm/bin/clang`, SHA-256
`570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888`.
I did not select it, rerun the failed gate, weaken leak checking or run affine
frontend parity or owned-record patterns in this attempt. A fresh corrected
attempt requires reviewed explicit compiler selection and new artifacts.

## I inspect the complete selector precedence before correction

I inspect the test paths without executing another fixture:

1. `CheckedOwnerSelection` binds `SourceBorrowEmission.execute_pair` directly.
   It does not run `SourceBorrowEmission.setUpClass`, so no emitter or shadow
   tool setup participates in this method.
2. `execute_pair` verifies and executes the module with the frozen
   `bin/nano_vm`, translates it with frozen `bin/nvm2c`, then chooses the native
   compiler with exactly `os.environ.get("CC", "cc")`.
3. An absolute `CC=/opt/homebrew/opt/llvm/bin/clang` would select the native
   compiler, but it would also be inherited by the three NanoLang compiler
   subprocesses that build the test drivers. I therefore do not use a global
   `CC` override when driver-byte identity matters.
4. Native execution copies the complete process environment and replaces only
   `ASAN_OPTIONS` with the existing
   `detect_leaks=1:halt_on_error=1`. The corrected selection does not disable or
   narrow ASan, UBSan, LSan, `-Wall`, `-Wextra` or `-Werror`.
5. The Homebrew installation contains both its Darwin ASan and LSan runtime
   libraries. I rely on the exact Clang23.1.1 binary and hash recorded above;
   I do not install or change any host tool.
6. The later `AffineFrontendParity` and `OwnedRecordPatterns` modules invoke
   only the frozen NanoLang compilers, NanoVirt, NanoVM and their produced
   binaries. They contain no host-compiler selector. The `CC` override will be
   scoped only to the affected checked-owner command.

The first attempt printed and sealed all three driver hashes, but
`CheckedOwnerSelection.tearDownClass` unconditionally cleans its
`TemporaryDirectory`. No retained `nano-checked-selection-*` directory exists,
so I cannot honestly reuse those driver bytes. The corrected method must build
fresh drivers and I require their printed hashes to equal, in producer order:

```text
3b0461984a3fca2f8a47bdb6b8d20bcaa6accfee66dff54319cf8f1abedbb337
91b64c860362c4e6069486db5168054a2bdd8edcb46450abe951b568d0f4cb14
1433157eb2c8916de42c37e86efbb4ea31e2886abfb72569c4de7c4b745553cf
```

I reserve a new detached exact603 checkout and evidence layout for review:

```text
/private/tmp/nanolang-parser-lsan-gate-603785c9
/private/tmp/nanolang-parser-lsan-evidence-603785c9
/private/tmp/nanolang-parser-lsan-tmp-603785c9
```

Before execution I will add a test-only `NANO_NATIVE_TEST_CC` override at the
single `execute_pair` compile call. Its precedence is
`NANO_NATIVE_TEST_CC`, then the existing `CC`, then `cc`. No producer, Make
rule or production compiler reads the new test-only variable. The corrected
command explicitly removes `CC`, so the three driver producers retain their
original environment while native generated-C compilation selects Homebrew
Clang23.

I will copy the eight hash-qualified compiler/runtime executables from the
preserved preceding checkout, freeze their hashes and the6,431 tracked sources,
then run exactly this affected method first:

```text
set -o pipefail
env -u CC NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
    TMPDIR=/private/tmp/nanolang-parser-lsan-tmp-603785c9 \
    /usr/bin/time -p python3 -m unittest -f -v \
    tests.test_checked_owner_selection.CheckedOwnerSelection.test_exact_lexical_dependencies_and_selected_suffix \
    2>&1 | tee /private/tmp/nanolang-parser-lsan-evidence-603785c9/checked-owner-selection-corrected.log
```

If it passes and all three driver hashes match, I will remove the command-scoped
`CC` override and run the two previously unrun modules separately, stopping at
the first terminal:

```text
set -o pipefail
/usr/bin/time -p python3 -m unittest -f -v tests.test_affine_frontend_parity \
    2>&1 | tee /private/tmp/nanolang-parser-lsan-evidence-603785c9/affine-frontend-parity.log

set -o pipefail
/usr/bin/time -p python3 -m unittest -f -v tests.test_owned_record_patterns \
    2>&1 | tee /private/tmp/nanolang-parser-lsan-evidence-603785c9/owned-record-patterns.log
```

I do not repeat bootstrap, the36-source matrix, callable controls, union
controls or any historical failed artifact. This updated checkpoint records
the inherited-driver-environment finding before the test-only selector change;
no corrected fixture has run yet.

## First selector-separated terminal does not reproduce driver identity

The first selector-separated method passed its complete semantic and native
LeakSanitizer checks in237.92 seconds. Its log SHA-256 is
`c1f5af4978fbac3e197f2b56f21208889c543de7eaf08352e7d182e07151e71f`.
That passing status is not yet qualifying evidence because the freshly linked
drivers were:

| Driver producer | SHA-256 | Prior sealed identity |
|---|---|---|
| C seed | `3b0461984a3fca2f8a47bdb6b8d20bcaa6accfee66dff54319cf8f1abedbb337` | equal |
| Stage1 | `7e9be553e24080e4c81191f6505b568de3671eb33220ab316def3d4152d34895` | different |
| Stage2 | `c21f93b2b8770017ba8e03d3e4a5fb86ac6c1111d21efc663e043ad4d00ea8ba` | different |

The eight copied compiler/runtime executable hashes remained the reviewed
ones. Static environment comparison finds one driver-build difference: the
failed Apple run inherited the host
`TMPDIR=/var/folders/9z/xpmfgw8j09l4g6wwxrxtt97w0000gn/T/`, while my first
selector-separated command replaced it with
`/private/tmp/nanolang-parser-lsan-tmp-603785c9`. The self-hosted native
publication route creates compiler and linker intermediates under `TMPDIR`;
therefore I do not infer a semantic regression or pretend the new bytes equal
the old ones.

I preserve this terminal and correct only the runner environment: the next
attempt keeps `CC` unset, inherits the original host `TMPDIR` for all driver
construction, and retains `NANO_NATIVE_TEST_CC` solely for generated-native
fixture compilation. I require the three original driver hashes before I run
the previously unrun affine modules. I do not repeat any already qualified
bootstrap, matrix, callable or union gate.
