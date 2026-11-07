# My generic ownership classification and layout admission

I reproduce #992 against main `6fc4bca18` in an isolated Darwin checkout.
The original12-case retained ownership matrix stops at `generic_cycle`:
current layout validation rejects its cyclic by-value generic union withE0038.
Commit60031f03b deliberately introduced that layout refusal; my native layout
controls already require it.

I give each row an explicit layout admission expectation independent from
its retained owned/collection expectations. All four original classification
comparisons remain, including generic recursion and resource collections.
Eleven valid rows still require acceptance. The invalid generic by-value cycle
requires refusal and exactly one cyclic-layout diagnostic withE0038.

The original fixture fails on the freshly built C compiler. The corrected
complete matrix passes with Clang and GCC16 on Darwin. Nine existing native
layout controls pass through `nanoc_c`: six cyclic-layout refusals preserve
prior output, and recursive array/list/generic-array layouts remain accepted.
I add the matrix to `test-quick` so hosted platform jobs cover this requirement.

This changes test expectations to match the established layout policy. It does
not remove ownership checks, admit recursive by-value layouts, or establish
full5.1 managed-runtime or exact-candidate qualification. Hosted checks remain
required before merge.
