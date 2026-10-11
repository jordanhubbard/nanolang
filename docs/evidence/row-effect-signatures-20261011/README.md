# Row effect signatures

I retain complete row-field annotations in my C parser, including nested arrays,
callbacks, generic arguments and row tails. I preserve those trees through copy,
substitution, nominal binding, equality, native type keys and serialized module
metadata. My metadata roundtrip also retains a nested service declaration origin.
My self-hosted lexer distinguishes `|` from `|>`, and my parser/checker validate
row declarations recursively without splitting nested commas.

I pass all nine compiler-product steps in `product-manifest.json`, with all 23
production inputs unchanged. My fresh seed passes 1,140 shadows. C, fresh seed,
and regenerated native compiler each pass 56 effect source cases: nine execution
cases, ten ordinary refusals, and 37 declaration controls. The regenerated
compiler also passes four effect-binding metadata methods and eight adjacent
opaque, union-storage and layout methods.

I pass parser, typechecker, effects and module metadata gates. My focused row
parse/copy/substitute/free control passes 100 iterations with ASan, UBSan and
leak detection. I retain the earlier private-parser-interface and bare-bar
failures in `before-interface` and `before-bar`. The probe is diagnostic only;
its root-only shadow selection is not release qualification.

These row cases validate declarations, not execution of row-valued operations.
I still require imported/nominal effect identity, resource ownership and native
handler execution. My previous broader parser leak report remains open. I do
not claim release qualification or close #976 or #989 from this evidence.

My audit script records absolute build paths for this Darwin run. I keep the
source hashes and commands so a later run can select its own scratch directory.
