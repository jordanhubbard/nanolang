# My closed record-recipe stack-height walk

Under #1142 I implement record_stack.nano over freshly decoded ARRAY/MAP
recipes. I reject reachable underflow, unequal-height joins/backedges and
incorrect explicit/implicit return heights. I measure both sides of reachable
instructions, include callee results in tail-exit maxima and retain the65535-slot
frame bound. Valid unreachable POPs do not invent a reachable error. The shared
height helper accepts an empty body independently of its declared result;
structural checking owns that separate refusal. This is not typed, lifetime,
origin, full-budget, whole-module or executable admission.

My thin C fixture includes the actual private recipe copier and shared verifier
translation units. It runs that copier as the closed-recipe precondition, then
uses the actual VM function decoder and verify_stack_heights. I do not clone
the height walk into an oracle or call it complete profile verification.

My first expected-positive MAP fixture encodes0x7c, HM_DELETE, while its comment
and expected height describe0x7f, HM_LEN. The actual ISA establishes that DELETE
needs an additional key. My original gate completes with exactly that one
expectation failure in186.603 seconds; every source-route report matches the
actual C refusal. I retain its exact source, carrier, ISA and complete terminal.
I correct only the opcode byte, preserving the positive maximum1 assertion.
No product/verifier source changes for that correction.

The corrected gate passes1475 reports per route,963 explicit expectations,
through all five Linux ARM64 routes in181.116 seconds: direct C-seed native,
both producers in NanoVM, and both producers in sanitized C AOT. I retain
linear and implicit exits, diamonds, backedges, invalid dead framing versus
valid dead underflow, nonzero code offsets, dynamic calls/literals, tail exits,
65535-depth maxima and actual frame overflow, plus512 randomized CFGs. Both
producers execute useful shadows under the existing component deadline.

The actual C copier/shared-verifier fixture also matches all1475 corrected
reports under targeted ASan/UBSan and leak checking. Other linked providers
retain their ordinary objects; I do not call this all-provider instrumentation.
The unchanged identifier gate passes five controls and3308 static fragments.

I retain actual failed/corrected derived products and complete inputs/reports,
source/tool identities, the original refusal and targeted sanitizer images.
My genuine retained native Stage1 artifact remains the NanoLang source provider;
its later-refused bootstrap receipt is not resumable authority. The separate
fresh c908523c3 bootstrap continues independently. My manifest covers3109 inputs,
188865409 raw bytes and37228373 unique compressed bytes with checked raw and
compressed SHA-256 identities.

Darwin, complete checked profiles, broader handler/effect/explicit-retain stacks,
value types, ownership/origins and pure translator/product cutover remain open.
I keep the paired roadmap row unchecked.
