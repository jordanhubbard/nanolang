# I qualify list removal without retiring pop coverage

I add self-hosted list remove/pop lowering and native `ARR_REMOVE` translation.
I preserve receiver aliases, record payloads, computed integer indexes, scalar
remove return values and the existing void result for generic record removal.
My source lowering checks signed bounds before mutation. Native removal retains
callable element flow and shifts closure environments alongside word storage.

My C-seed-built emitter driver completes its build and shadows. Before native
removal support, the paired eight-method suite is not complete: the earlier
seven-method run records twelve failing subcases. After native removal support,
the eight-method run records six failing subcases, all at unsupported native
`ARR_POP`. I retain those required pop cases, including empty-pop, unchanged.
Removal without pop and the inherited insertion controls pass through both
producers, VM execution and strict sanitized native execution. Operand refusals
and removal bounds controls also pass. The remaining pop failures prevent full
list-removal qualification; they are not expected-success tests.

The input hashes identify my dirty source checkpoint and executables observed
after the component run. The subsequent Make regression may rebuild tools;
these hashes do not establish an immutable pre-run binary manifest. I retain
the component sources, bytecode, assembly and command logs.
Fresh installed compiler products, the complete paired parser corpus, raw
record-pop empty-result semantics, Linux qualification and final 5.1 gates
remain open under [#978](https://github.com/jordanhubbard/nanolang/issues/978).

My subsequent `make -j2 test-nvm2c CC=/opt/homebrew/opt/llvm/bin/clang`
terminates with exit zero: 2,431 execution, 3,092 shape and 379 callable checks
pass, along with the target’s Python controls. This does not close the six
required pop failures in my component suite.
