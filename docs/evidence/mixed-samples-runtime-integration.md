# My canonical mixed runtime integration

I preserve the qualified tree at2265707c3 and integrate canonical88e7dc746 in a
fresh worktree at45d9e4e41. Only appended Makefile targets and roadmap history
conflict; I retain both sides. My [manifest](mixed-samples-runtime-integration/manifest.json)
compares every production source/script changed by this feature against the
qualified commit, with identical bytes. Incoming production consists of private
File/Socket/GPU/service-codec foundations and the two canonical mixed-provider
closure additions. Required-service transport guards are not yet canonical.

Fresh tools build in17.859seconds. The full12-case GCC native/runtime, VM heap,
admission and private-query integration gates pass in36.758seconds; raw service
codec and its sanitizer adjacency pass in0.467seconds. Runtime2006, heap13358
(488budgets/440failures), admission127, private112482/339 and strict native
O0/O2 sanitizer/allocation assertions remain intact. I do not claim a new compiler
bootstrap, repeated fullnative gate, new Clang result or Darwin qualification.
Their earlier measured scopes remain in the original seal.

All tracked sources agree before/build/after; actual built tools and linked object
inventories agree before execution and after all gates. No source producer changes,
owner managed-field admission or full-parent closure follows. The later service
transport integration must preserve service refusal ahead of mixed admission and
qualify combined metadata retention; I coordinated this boundary with its owner.
