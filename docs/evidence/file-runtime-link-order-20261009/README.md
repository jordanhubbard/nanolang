# I order File test consumers before archive providers

Under #982/#989 I retain the Linux arm64 failure from CI run38017788965 at
`a142b6018`. The corrected fixture compiles, but GNU ld cannot resolve the
File snapshot/plan/catalog functions needed by `env.o`, `module.o` and
`service_namespace.o`. The retained archive-build excerpt confirms those
providers were compiled and archived. The exact failing link command places
`libnano_compiler_inputs.a` before those consumers.

My File runners deduplicate inputs, removing the archive occurrence after the
common objects. Darwin's linker accepts the earlier order; Linux's one-way
archive scan does not. I define shared `FILE_RUNTIME_TEST_OBJECTS` in Make:
all explicit objects precede the existing archives. Eighteen File runtime,
frame, public and dispatch recipes use that list. I preserve every input and
the archives' original relative order; I add no force-load or whole-archive flag.

I extract both old and corrected Make lists and apply the same deduplication.
Their membership is identical, and every archive follows all explicit objects
in the corrected list. A small AArch64 ELF consumer/provider fixture uses the
actual relative order of `obj/env.o` and `libnano_compiler_inputs.a` in those
lists. LLD with `--warn-backrefs --fatal-warnings` rejects the old backward
reference and accepts the corrected order. This checks ELF link ordering; it
does not run a Linux executable or substitute for the complete Linux CI gate.
Commands, diagnostics, exit codes and both expanded lists are in
[the result](elf-results.json). I retain the reproducible extraction Makefile
and fixture script beside it.

My first extraction attempted Make's `file` function, which this local Make
did not implement; no list files were written and the ELF script stopped with
`FileNotFoundError` before linking. I use portable `printf` recipes instead.
The corrected ELF comparison then passes. This setup failure is not a product
or linker failure.

The actual corrected Make carrier target passes both instrumented and linked
methods in12.048 seconds. I retain its full log. All five workflow checks pass,
as does `git diff --check`. The neighboring callable dispatch corpus was already
qualified with its recorded Darwin link inputs; I do not relabel those earlier
links as a Linux run. Exact-revision remote execution remains open.
