# Scalar reduce and global array result storage

I resume #976 from cfa60d6eb. I replace the obsolete Boolean ABI refusal with
executed true, false and empty reductions, retaining invalid-signature
prior-output checks. `NANOLANG_REDUCE_COMPILERS` selects explicit compiler paths;
my default still qualifies installed Stage 1 and Stage 2.

The unchanged six-method suite passes current products (`current.log`).
Extending ordered operand evaluation to canonical NanoISA exposes a current
C AOT failure (`ordered.log`): the C-produced module verifies and executes in
NanoVM, but a helper returning a global float array has no supported native
result storage. `ordered.nasm` retains that module's assembly.

I discover all constructor-backed global array storage before ordinary
inference, then preserve the exact input storage hint across checked ARR_PUSH
and ARR_SET. Storing that result back to a global retains its proven kind.
I do not infer an array type from an arbitrary tagged store. Runtime tag checks
remain in place. My conflicting int/float global-return control verifies and
runs in NanoVM but remains refused by C AOT without replacing prior output.

I add int, float, bool and string global-return controls with empty values,
push, replacement and observed results. The first fixture incorrectly assigns
array_set's void result; both frontends reject it (`mutation.log`). The corrected
fixture executes the mutation as a statement (`mutation-final.log`).

My old-installed-stage run (`installed.log`) records four ordered-shadow
failures plus the then-unfixed C AOT failure. I retain those results; current
compiler selection does not establish an installed-stage or platform release.

I retranslate the pinned compiler module from the previous compiler-product
audit and link a new native compiler. `product-manifest.json` records seven
successful steps, including 16 nominal methods, the source suite and 16
payload/resource/layout methods, with unchanged translator/module inputs.
My final eight-method source suite passes with the fresh seed and retranslated native compiler (`final-source.log`), including four scalar array element kinds, verified VM execution, sanitized C AOT, compiled Boolean `--target c` output and prior-artifact refusals.

The full 5.1 scope and #976 remain open. GitHub API access still fails;
I preserve my issue update and error rather than claim it was posted.

My broad `make CC=/opt/homebrew/opt/llvm/bin/clang test-nvm2c` gate passes all prerequisites and 2,443 structured C tests (`aot-gate.log.gz`).
