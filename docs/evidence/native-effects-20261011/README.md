# Native effect activation and unwind

I lower my verified handler registrations, performed operations and resumptions
into native C activation frames. I keep each arm's arguments and temporary
locals separate, while earlier locals refer to their lexical owner. Ordinary
C returns propagate a lexical exit through intervening calls; each frame
releases its temporary storage and registered roots. I prune its handlers on
return and tail replacement. I do not embed bytecode or invoke a VM.

I infer operation argument/result representations and callable provenance
across arm entries. I share arm reachability between my representation and
callable analyses, including backward entries. Explicit RET exits the lexical
owner; EFFECT_RESUME and implicit one-result completion resume the performed
operation. Registrations retain their own argument counts even when they share
an entry label. In modules using this activation runtime, HALT propagates to
main, preserves the observable integer stack top and performs normal cleanup.
I isolate handler lookup across emitted foreign-call boundaries; actual foreign
callback re-entry is not qualified by these tests.

I also parse C handled bodies as expression blocks and prevent their local
bindings from escaping. This closes the paired statement-block/escape controls,
not the broader imported effect identity or ownership requirements.

## Verification

- I pass 66 source cases through the C frontend, fresh self-hosted seed and
  regenerated native compiler. Eighteen programs per producer run in both my
  VM and native C; eleven refusal cases preserve prior artifacts; 37 controls
  check effect declarations. My final paired run instruments the translator,
  callable/shape analysis objects and generated native programs with ASan and
  UBSan, with leak detection enabled.
- I pass ten raw boundary methods (sixteen fixtures) with the same sanitizers.
  These include expected traps, private activation locals, callable transport,
  lexical cleanup, backward entries, implicit completion, reused entries and
  HALT cleanup. Native traps must be SIGABRT with my invariant diagnostic;
  sanitizer failures and arbitrary crashes do not count as expected traps.
- My broader native gate passes 2,443 core checks, 464 callable constraints,
  3,113 shape constraints and its adjacent Python suites. Parser, typechecker
  and effect component gates also pass (`regressions-llvm.log`). My initial
  broader run used Apple's compiler, whose sanitizer rejected leak detection;
  I retain that run and the corrected LLVM run without disabling the check.
- My nine-step compiler-product audit passes with all 29 tracked inputs
  unchanged. My fresh C-built seed executes all 1,140 shadows with zero failures.
  The regenerated executable passes the source cases, four effect metadata
  methods and eight opaque/union/layout methods.
- After that audit I correct shared-entry counts and HALT propagation, and make
  growth of result-tag facts restart inference. My final
  translator reproduces byte-identical compiler C from the audited module;
  `final-retranslation.json` identifies every changed input and final hash.
  The final `paired-tags-sanitized.log` and `boundaries-tags-sanitized.log` runs
  use these corrected sources; the broader core run precedes the final
  result-tag convergence correction.
  My six executable/runtime tool inputs remain unchanged throughout those
  final checks. The compiler module also matches my previous row-signature
  module byte for byte; this is not an exact release-platform qualification.

I retain the initial callable-analysis refusal, strict-C helper warning,
backward-entry failure, malformed handled-body refusal and the two late raw
control failures. Corrected tests preserve their original semantic assertions.

## Reproduce

```sh
NANOLANG_EFFECT_COMPILERS=bin/nano_virt:/private/tmp/nl51-native-effects/seed:/private/tmp/nl51-native-effects/compiler \
NANOLANG_EFFECT_TARGETS=vm,native \
NANOLANG_NVM2C=/private/tmp/nl51-native-effects/nvm2c-asan \
NANOLANG_EFFECT_CC=/opt/homebrew/opt/llvm/bin/clang \
NANOLANG_EFFECT_CFLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -g' \
ASAN_OPTIONS=detect_leaks=1 python3 tests/test_selfhost_effects.py

make CC=/opt/homebrew/opt/llvm/bin/clang test-native-effects
```

My retained scripts use this Darwin checkout's absolute paths. I verify actual
ASan/UBSan symbols in my translator, callable and shape objects. Other linked
objects are not instrumented by that focused translator build.

## Remaining acceptance

I keep #976 open. Affine resource effect payloads, complete imported/nominal
signature identity, mixed tagged/scalar parameter carriers, real foreign
re-entry and exact Linux/Darwin release qualification remain required. My
ordinary HALT probe also finds that a module without handler opcodes still uses
the older local-return lowering: VM exits zero, native continues main and exits
99. I retain this separate uncovered path in `ordinary-halt.json`; I do not
claim all HALT behavior complete from the effect-module controls.

My worker claim is published at coordination commit
`a93d419195b21781b3ac06d4870a6ea1024636fc`. GitHub API posting failed in the
retained attempt. Successful Git transport does not imply that my issue comment
was posted, and I do not contact MAC.
