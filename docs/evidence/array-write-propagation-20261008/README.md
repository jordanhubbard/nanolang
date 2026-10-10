# My shared-array write propagation

I separate actual `ARR_PUSH`/`ARR_SET` writes from callee read views. Direct
array parameters register handle aliases. Conversions discover nested array
aliases inside copied records. My solver carries written values back through
those aliases and retains checked optional/exact field views. A joined callee
read view no longer becomes a write to every unrelated caller.

My holder-record reproducer now belongs to the production mutable-record-array
suite. The suite covers direct, helper, forwarded, holder and indirect mutation
with both push and set through both producers: 20 source products, plus the
existing diagnostic formatter through both producers. VM output must match
strict C11 native execution with ASan/UBSan/LSan. Both the preceding native
compiler and the newly translated native compiler pass these 22 products.

I pass 2,431 native checks, 3,092 shape checks (also under ASan/UBSan/LSan), and
379 callable checks. Added graph controls retain unrelated producer function
targets, propagate late writes across nested aliases, survive later exact joins,
converge through 128 cyclic aliases, and remain stable when solved again.

My first actual-write prototype scans every alias/write pair repeatedly.
It passes the focused source and native tests but runs the retained compiler
translation for 358.942 seconds before I intentionally stop it with SIGTERM.
I retain its patch, CPU sample and explicit stop receipt; this is not a natural
terminal test failure. The sample places most time in pair rescans.

I replace those scans with insertion-key hash sets and per-pass reverse
adjacency rebuilt after exact joins. Later joins may leave equivalent insertion
keys; root-resolved traversal preserves their meaning and repeated solves
converge. The indexed translator emits the same retained compiler module in
10.921 seconds. Strict C11 compilation succeeds, and that new compiler passes
the expanded source matrix. I retain input hashes and command/result evidence.

Commands use `BIN_DIR=/private/tmp/nanolang-write-indexed-tools/bin` and
`OBJ_DIR=/private/tmp/nanolang-diag-tools/obj` for Make `nvm2c`, `test-nvm2c`,
`test-nvm2c-shapes`, and `test-nvm2c-callables`. The source suite is
`python3 -m unittest -v tests.test_native_mutable_record_arrays`, with
`NANOLANG_TEST_NVM2C` selecting that translator, `NANO_NATIVE_TEST_CC` selecting
Homebrew LLVM, and `NANOLANG_SELFHOST_COMPILER` selecting the retained or new
compiler. The sanitizer graph build uses Clang with address/undefined checking
and `ASAN_OPTIONS=detect_leaks=1`.

My complete compiler-product gate at `1321b8bdf` passes all 109 methods.
The primary checkout retains unchanged HEAD, a clean tracked tree, and only
the same user-owned untracked test, whose hash is unchanged. I archive the
full log, runner and terminal receipt; this is not a claim of an entirely
clean checkout. Integration, clean final-candidate gates, Linux acceptance,
and the remaining release scope stay open. The separate full source-snapshot
rerun remains pinned to `f0f6a0c62` in its own checkout.
