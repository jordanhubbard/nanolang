# OPL library bytecode coverage

I removed `opl/opl_codegen.nano` and `opl/opl_compile.nano` from my VM
exclusion list after repairing their fixture paths. Their shadows now pass
and both libraries lower to bytecode. They remain import libraries in my
native example selection; I have not added standalone native entrypoints.

At base `40799db0`, with this selection change on Linux arm64, I ran
`make stage1` followed by `make -j8 test-vm-examples`:

- I accounted for all 248 example sources: 244 eligible, four excluded.
- All 244 eligible examples compiled to bytecode with shadows enabled.
- Each of the four exclusions still failed with both VM and native compilers.
- My 400-level nesting case compiled and executed with result 400.
- My temporary-file and GPU host-geometry builtin checks passed.

The gate derives these counts from the example tree and Makefile selection;
I did not change it to assume a fixed count. Earlier evidence reporting 242
eligible sources records the earlier selection and remains unchanged.
This is bytecode compilation coverage, not execution of every example.

Log: `/tmp/nanolang-opl-vm-coverage-final.log`.

I also corrected the standalone gate prerequisite to depend on my existing
`$(COMPILER)` target. Three dependency shadows invoke `bin/nanoc`; building
only `bin/nanoc_c` did not create that alias in a fresh checkout. The existing
target preserves my C-seed/selfhost selection policy. After removing the
owned worktree's generated alias, `make -j8 test-vm-examples` recreated it
and passed the complete 244-example gate again.

Standalone log: `/tmp/nanolang-opl-vm-standalone.log`.
