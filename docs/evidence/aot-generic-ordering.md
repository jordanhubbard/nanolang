# My native generic ordering

My VM's `val_compare` does not require matching integer operands. Matching
integers and booleans compare their values; strings compare lexically. Unlike
tags order by tag number, apart from explicit numeric cross-type cases. Arrays
and maps of the same tag compare as zero, not by their contents.

I now lower `LT`, `LE`, `GT` and `GE` using those rules for my supported native
representations. Scalar and tagged operands retain their tags. Native array
representations share the array tag; native maps retain the map tag. I preserve
the boolean result tag through classification and emission. Integer ordering
uses relational comparisons, not overflow-prone subtraction.

I do not reinterpret floats as integers or guess erased record/union tags.
Float, enum and other unpreserved representations remain outside this native
subset. This is not complete generic-comparison support for every VM value.

## Verification

`make -j1 test-nvm2c` and `make test-nvm2c-sanitizers` each pass 1,444 AOT and
994 shape checks. The matrix covers all four opcodes with integer extrema,
equal and unequal values, booleans, lexical strings, unlike tags, void-before-
store, tagged global scalars, arrays and maps. Each result is checked as a
boolean as well as for its truth value. Generated C compiles with warnings as
errors. The sanitizer run verifies fresh ASan/UBSan translator and shape
objects; opcode and sanitizer-driver checks also pass.

The full compiler gate remains failing. Fresh bytecode passes generic `LT` in
function 532 and reaches function 569, `exists`, whose string argument is
rejected at `CALL_EXTERN 29` (`vm_file_exists`). Its body is a local load,
the host call and return. I keep exact host ABI validation and track tagged
argument consumption as the next boundary to resolve.
