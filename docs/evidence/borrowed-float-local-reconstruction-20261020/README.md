# I retain borrowed FLOAT caller-local reconstruction

I extend only ownership-v2, two-function, single-exclusive modules with caller-only nonparameter FLOAT/NO_INDEX locals. PUSH_F64/F64_NEG/F64_EQ are the bounded new operations; fields, signatures, helper FLOAT and FLOAT pretests remain excluded. No runtime, verifier or frontend changes.

The unchanged ordinary and independently captured original-shadow baseline first passed facts but refused Nano output at 131/21 locals, preserving prior output. Existing caller-only finite planning now regenerates both at actual 98/21. It preserves ordered FLOAT expressions, immutable input IR and the prior rendered helper, uses no scratch bank, and refuses overlapping helper-local/reassignment/exit profiles.

The six-method corpus against seed and both retained fresh stages passed five methods; the generated-shadow method initially attempted to reconstruct the generated witness again and hit the existing single-root reconstruction restriction. I preserve that terminal (51.100 seconds). The corrected test qualifies the selected generated witness itself through all three compilers, verifier, VM and sanitized native execution, and passed in 3.424 seconds. All six intended methods are green across these terminals. I do not claim generated multi-root witness re-reconstruction support.

Coverage includes unchanged ordinary/original-shadow C and Nano outputs, O0/O2 ASan/UBSan execution, negative-zero equality, exact 128 acceptance/129 atomic refusal, forbidden FLOAT operation/helper slots, original IR and ordered FLOAT preservation. Eight previous INT/BOOL C/Nano outputs remain byte-identical. GCC16 and Clang strict facts syntax pass (an initial command omitted the existing modules/nanoisa include path; corrected commands use the real include paths). Ten provider/source hashes match before and after.

This is local component evidence using existing stages, not a new raw bootstrap or hosted Linux/Darwin qualification. The complete aggregate and platform gates remain required.
