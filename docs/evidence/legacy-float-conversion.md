# My aligned interpreter and C-emitter float conversion

I use the NanoISA float-to-int policy in my interpreter and both legacy C emitters: finite values in `[-2^63, 2^63)` truncate toward zero; NaN, infinities and values outside that interval stop with the same explicit conversion diagnostic before the integer cast. Integer-to-integer conversion remains separate and unchanged.

The first real source fixture exposed a missing C-seed helper: `string_to_float` was registered but absent from emitted runtime C. I retain the failed compile log and add that helper using the existing interpreter/self-hosted `strtod` contract. I reran the original fixture, then expanded its string parsing cases to empty, invalid and trailing-text inputs.

At source checkpoint `6f7a2ca6`, `make test-legacy-float-conversion` completes a fresh three-stage bootstrap and passes 64 conversion cases across the interpreter, C-seed compiler, Stage 1 and Stage 2. The three compiled outputs use UBSan with float-cast-overflow instrumentation and non-recovering errors. Valid cases include both truncation directions, signed zero and representable interval endpoints; rejected cases retain the explicit diagnostic without sanitizer errors. The final unittest completes in 11.333 seconds after bootstrap. This does not claim unrelated interpreter or compiler behavior is proved.

Task `task_f801bf5769f9489da5ea973574dd156c` owns alignment; `task_5909147f37c2478a8c07494935d7e24a` owns the missing C-seed helper. Evidence remains in `/tmp/nanolang-legacy-float-tests.log` (initial missing-helper failure), `/tmp/nanolang-legacy-float-tests-repaired.log`, and `/tmp/nanolang-legacy-float-final-gate.log` (fresh bootstrap and expanded cross-stage gate).
