# My effect ownership declaration evidence

I add the version-3 EFFECTS extension and atomic operation-signature query described in [my contract](../../NANOISA_EFFECT_OWNERSHIP.md). My ownership transport/execution gate passes; the instrumented codec and test harness pass 888 checks under AddressSanitizer and UndefinedBehaviorSanitizer with leak detection. Other linked objects retain their existing build instrumentation.

I test owned records, distinct nested owned-union arguments/results, scalar/string/void boundaries, zero and maximum parameter counts, maximum operation count, duplicate keys and string identities, reserved flags, wrong layouts, every payload truncation, and unchanged query output on error. Binary and unverified assembly transport round trips preserve the payload. Verified assembly, NanoISA verification and native translation still refuse the unimplemented affine effect instructions.

The initial test mistakenly required verified assembly to accept those instructions. I retain initial.log and roundtrip-diagnostic.log; the corrected test requires that refusal and uses the existing explicit unverified assembly API solely to test transport. I do not weaken executable verification.

Adjacent gates pass: 413 affine-state checks and 445 with allocation faults; 1,150 affine-bytecode checks and 1,787 with allocation/visit-limit faults; ordinary-array and mixed declaration projection, including their fault-injection checks. inputs.json pins the changed implementation/test inputs.

This checkpoint supplies declarations. Producer wiring, instruction transitions and runtime ownership are still required; I do not claim resource-effect execution or full 5.1 completion.
