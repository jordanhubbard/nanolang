# Ordinary source names and canonical function layout

I retain source parameter and ordinary let names for scalar, array, record,
tuple, enum, union and callback values in both frontends. Explicit generated
binding flags omit destructuring owners while retaining user names such as
`__td99`. I preserve the flag through parser copies and regenerate the extern
AST schema with matching Python and fresh NanoLang generators.

I fixed two canonical assembly defects exposed by these tests. I retain dots
in generic function identifiers so `$generic.1` and `$generic.2` remain distinct.
I publish C-produced function bodies in function-table order, relocate absolute
DEBUG positions and then publish passive metadata. Function-relative local
intervals and within-function branches stay unchanged.

## Verified checks

- My eight-method suite passes both the rebuilt C producer and fresh self-hosted
  seed. It checks exact original module bytes and second-cycle assembly bytes,
  lexical intervals, generic parameters, imports and generated destructuring.
- My combined initializer/lambda/imported-parallel fixture checks execution and
  canonical layout across different function-emission orders. My imported
  source fixture explicitly checks the relocated first DEBUG source line.
- Removing only local-name metadata preserves VM results and byte-identical
  generated native C. Normal native products execute with the expected results.
- I pass 90 bytecode tests, 273 passive checks and 26 passive Python tests,
  210 disassembly roundtrip checks, parser and typechecker component gates.
- My fresh seed passes 35 callback, generic, union and tuple regression methods.
- My fresh schema generators produce identical hashes of all four outputs.

I retain the original duplicate-generic-symbol and imported-wire failures, plus
an initial shadow-test setup failure caused by calling a private parser helper.
The corrected shadow parses an actual destructuring declaration.

My fresh native compiler product passes all seven recorded audit steps in
`product-manifest.json`: self-emission, verification, wrapper-free C translation,
strict native linking, CLI execution, all eight local-name tests and 22
nominal/layout regressions. All ten source input hashes remain unchanged. This checkpoint does not establish
closure/upvalue naming, all frontend facts, full reconstruction or the exact
Linux/Darwin release-candidate gates. Those requirements remain open under #976.
