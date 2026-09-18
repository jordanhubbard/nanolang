# My generated array graph acceptance

I qualify the bounded generated part of task4070da262. Private runtime childc01
landed in PR729; full aggregate488 and managed51da remain open.

My pre-code contract is `42a2d57f`. Production `dfc4fa68` received independent
review; the only subsequent graph production change asserts MEMORY status3.
Frozen final source/tests `45e46da4` passed the full gate below. I restacked onto
arithmetic PR731 at `c8db7cc5`, yielding `33d973cd`. Git range-diff reports all
five graph commits equivalent; graph selector/header, managed helpers, verifier
and graph tests are byte-identical. The LLVM emitter additionally contains the
independently reviewed scalar arithmetic normalization from731.

## My accepted boundary

I select existing leaf mode first. Only UNRESOLVED may try the separate graph
query; failed allocation, limits and invalid input remain final. A selected graph
uses qualified counted roots, prepared pre-allocation collection and final
collection after frame cleanup. Static origins do not infer liveness.

I retain stack/local/global and transferred call/result owners; GET retains and
POP transfers child ownership. Nested literals and shallow slices preserve
children and fresh outer identity. Preparation MEMORY occurs after entry
acquisition and reaches graph finish; BUSY/DISPOSED does not acquire or finish an
entry. First errors survive cleanup and completed global writes persist.

## My observed gates

I set `NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.

- Frozen `make -j4 test-llvm-managed-strings test-managed-array-eligibility
  test-verifier-profiles` passes: 69 managed methods in 94.556 s, 27 shape
  methods in 3.510 s, four graph/prepared-runtime methods in 6.686 s, plus
  profile, core and exact package generation/linkage checks.
- After731 integration, a fresh `make -j4 nvm2c nvm2wasm nanoisa_dump nano_vm`
  and eight graph plus two arithmetic methods pass (10 methods, 15.340 s).
  The two arithmetic methods are `ManagedArithmetic.test_managed_scalar_cleanup`
  and `ArithmeticBackends.test_mixed_promotion_and_input_transport_unchanged`.
- Graph tests use normal assembly/verification and public LLVM/Wasm translation,
  VM reference execution, ASan-instrumented generated native IR, import-free Node
  and Wasmtime. They test nested aliases, counted literals, slices, GET/POP,
  returned temporary roots, prepared split-child insertion and replacement,
  first errors, initializers, global identity, disposal and repeated entries.
- A generated loop creates 12,000 mutual/self cycles with string children per
  entry under a 1 MiB Wasm limit. Repeated successful entries end with zero live
  objects. A separate live packed child reaches real Wasm allocation pressure,
  returns MEMORY while keeping exactly two global-reachable objects, and clears
  both at disposal. Fresh instances repeat the same observed behavior.
- Fresh native executions independently fail allocation budgets0,1,2,3,4,6,10,
  15,25,45,80 across preparation and object/table/workspace/literal/slice growth;
  every acquired failure clears active frames and later corrected entry succeeds.
  Nested begin retains the active caller's original ASSERT error; terminal
  reentry refuses. Leaf mode stays unprepared.
- Shared selector probes preserve output on failure, including allocation budget0;
  incompatible packed children, unknown shapes and existing nominal/import
  boundaries remain refusals. Newly supported exact graph cases now execute;
  I do not merely remove their prior refusal assertions.

## My retained development evidence

The first build omitted the established host Clang GCC-directory flag and stopped
on its installation-selection warning. The corrected explicit-flag build passed.
The first new global-cycle fixture was rejected during assembly because an extra
DUP left incompatible branch stack heights; it was corrected before execution.
The first expanded profile and then 66/68 managed result retained old intentional
nested-array refusal expectations. I moved those exact now-supported cases into
paired positive execution and retained adjacent packed/unknown refusals. The
corrected affected ten methods passed13.442 s before the final full run.

I retain representative ordinary cycle assembly/module, native LLVM/executable,
Wasm and actual execution/tool hashes under
`/tmp/nanolang-graph-admission-artifacts-33d973cd`; the committed manifest records
all commands and hashes. These are bounded artifacts, not full compiler bootstrap
or Darwin/platform/release acceptance. No historical crash artifact was replayed.

## My retained logs

- `/tmp/nanolang-graph-admission-build.log` — SHA256 `915bc166cb8365815b90be3a5d0e48b34bc21a903bff087a9ebc063d33f32ada`.
- `/tmp/nanolang-graph-admission-build-corrected.log` — SHA256 `a0f5f85931c0660f95f65ed5fd100df48043fb86950171c8bcda85610acdb6e6`.
- `/tmp/nanolang-graph-admission-focused.log` — SHA256 `0a7639db9bc6590aea0741f30be36ee7fba0c8e4a0861306f9be8a5fdb592a64`.
- `/tmp/nanolang-graph-admission-adjacent.log` — SHA256 `b3cb7e9f9e55d4ee167a37cf3756ac471be39dd842ed9c250092259946c101e0`.
- `/tmp/nanolang-graph-admission-adjacent-corrected.log` — SHA256 `340337401752fcab37d897c1e6b6c09b178e6656d72135d5a47c854ae1c79468`.
- `/tmp/nanolang-graph-admission-refusal-transition.log` — SHA256 `513725251b241413fe4ef9c06fa49b62b4b24e1ee1dd5c1cb3ad1e4fe9b4514c`.
- `/tmp/nanolang-graph-admission-final.log` — SHA256 `c53dbbcdcf16799787c169f09a79838262b9eeb5543738245b461d85092c90b7`.
- `/tmp/nanolang-graph-admission-integrated-build.log` — SHA256 `e48ff557f796c90ce0a13ff3d9da8335e113ceaa0e3b090015f7dc21e1d19900`.
- `/tmp/nanolang-graph-admission-integrated.log` — SHA256 `b9f6fab48ac18af8f7b0131ab0567fdb25b4f503385e1081b7f37aaa9b7d2cf2`.
