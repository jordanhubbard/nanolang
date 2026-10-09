# My C-seed public-global import checkpoint

I bind imported spellings to the checked declaration, retaining its type, mutability and runtime storage identity. I resolve aliases in their importing source context and preserve local and captured-variable precedence. Qualified immutable scalar reads participate in purity checking; mutable reads remain refused in pure functions.

I retain four failed checks: the initial selective-alias context and purity failures, captured-local precedence, legacy plain-import constant visibility, and legacy native literal folding. I correct those paths without copying imported bindings into a second runtime global.

My legacy plain imports continue exposing immutable constants without `pub`. Qualified and selective imports still require public declarations. This compatibility exception qualifies the visibility requirement in #986.

My corrected C-seed suite passes eight methods in 2.578 seconds. A freshly seed-built source compiler, compiled with dependency and root shadows, passes those same eight methods in 4.576 seconds. Successful products execute in NanoVM and strict C11 native code with ASan/UBSan and leak detection. Refusal cases preserve prior output. The adjacent checks pass 52 environment assertions, ten lexical-scope methods, all C typechecker tests and three module-global identity methods.

I run the adjacent C checks with `make -o stage1 test-env-scoping test-typechecker` after rebuilding `nano_virt` and `bin/nanoc_c`. This does not establish a fresh bootstrap. I retain exact logs and changed-source hashes.

I still require broader canonical-path, wildcard/conflict, initialization, callable/managed ownership and qualified-write checks; freshly installed Stage 1/2; Linux/Darwin acceptance; and a new full raw fixed point. #986 remains open. GitHub API and SSH DNS access fail during this checkpoint, so my issue update and push remain pending until remote access returns.
