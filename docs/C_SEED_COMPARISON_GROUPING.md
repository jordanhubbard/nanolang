# I preserve binary operand grouping in my C seed

My native C emitter previously placed nested comparisons beside their parent operator without grouping each operand. Strict GCC rejected the existing comparison regression with parentheses diagnostics. I now wrap each operand as my Nano emitter already does, preserving the existing outer-parenthesis policy and operator selection.

At b4ddc1edb, fresh builds and three-stage bootstraps pass on Linux and Darwin. All six array compatibility methods pass, including the unchanged 16-pair comparison assertions through C seed, Stage1 and Stage2. The existing Boolean-precedence program also compiles and executes through all three producers on each host. Independent source review passed.

I retain the [evidence seal](evidence/c-seed-comparison-grouping/seal.json), original generated C and failure, exact runner, commands, phase statuses, logs and source/tool/product hashes. Larger report files are losslessly gzip-compressed; the seal names both original and stored hashes. Retained CAS stores remain at their recorded paths; the downloaded Darwin archive hash was verified. Inner temporary outputs removed by the existing unittest are not reconstructed. The separate Boolean binaries and generated C are retained.

I do not infer the NanoISA-only bootstrap fixed point or whole release readiness from these native compiler gates.
