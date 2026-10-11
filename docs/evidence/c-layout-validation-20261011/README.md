# C frontend declaration layouts

I close the C frontend declaration-layout gap under GitHub #976. I validate
root and imported record/union declarations after registration and before
field-instance publication or backend emission. I follow substituted union
payload annotations and compare complete nominal applications. Arrays, lists,
callbacks and other pointer-backed types do not create by-value edges.

I distinguish an active by-value cycle from expansion beyond 128 nested
generic substitutions. Ordinary record/tuple traversal does not consume that
budget. I keep caches local to each declaration's generic formal scope and
restore the module context after each traversal.

My shared frontend contract checks six cycles, an expanding unused template,
equal record names from different modules, preserved prior native/NanoISA
artifacts, legal indirect/unused generic fields and a 145-record acyclic chain.
I invoke nanoc_c for native output and nano_virt for NanoISA; I do not call
packaged interpreter output native AOT.

Validation results are recorded alongside this document. These checks do not
close aggregate payload lowering, the NanoISA-only compiler transition, or
exact-candidate Linux/Darwin qualification.

I pass 5 shared C layout methods, 57 mixed native/NanoISA/VM/C AOT regression
methods, 41 explicit native regression methods and make test-typechecker.
The 21 sanitizer-assisted methods instrument typechecker.c with Homebrew
Clang ASan/UBSan for native compilation; the other linked compiler objects
and nano_virt remain ordinary builds. I do not claim whole-compiler sanitizer
coverage. The regression suite also executes generated C AOT under sanitizers.

My GitHub API status query fails with a connection error; I retain its output.
Git HTTPS confirms my ownership claim on the coordination branch at
4464644da4d4ce311e36f11e56f47796de4f859f.

I reproduce the focused checks with:

```sh
make -j4 bin/nanoc_c bin/nano_virt
python3 -m unittest tests.test_c_layout_validation -v
NANOLANG_NOMINAL_COMPILERS=nanoc_c python3 -m unittest tests.test_native_nominal_order tests.test_module_union_owners tests.test_generic_union_parameters tests.test_cseed_generic_functions tests.test_module_function_names -v
NANO_CONTEXTUAL_NATIVE_COMPILER="$PWD/bin/nanoc_c" python3 -m unittest tests.test_module_union_owners tests.test_generic_union_parameters tests.test_cseed_generic_functions tests.test_module_function_names -v
make test-typechecker
```

My retained sanitizer helper consumes `make -n -B bin/nanoc_c` output in
`/private/tmp/nl51-c-layouts/build-recipe.txt`, replaces only the checker object,
and writes a separate executable without relinking the active compiler.

I compile `src_nano/nanoc_v06.nano` with the corrected nanoc_c and run all
five self-hosted layout methods through that fresh seed successfully. I do
not rerun the eleven-step native product audit for this C checker-only change.
