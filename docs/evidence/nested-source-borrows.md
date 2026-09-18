# My nested source-borrow acceptance

I implement MAC `task_91cb8c2db94941bc950d9e9fa597d77a` from main `d0f72962`.
Contract `91d1e746` precedes production checkpoint `f0a9fdb8`. My
[bounded contract](../NANOISA_NESTED_SOURCE_BORROWS.md) admits declaration-ordered
finite owned trees and exact numeric caller paths. My runtime, verifier,
retained-layout schema and ownership formats are unchanged.

I measured these gates on that production source:

- A fresh default three-stage bootstrap passed:
  `/tmp/nanolang-nested-source-bootstrap-r2.log`.
- All 12 paired source-borrow methods passed in 157.954 seconds:
  `/tmp/nanolang-nested-source-paired.log`. I reused the completed bootstrap
  with `make -o bootstrap test-source-borrow-emission`; I skipped no test.
  The target also passed 123 lexical-name checks, five name-allocation
  boundaries and twenty marker-allocation boundaries.
- My unchanged nested-reference and multi-caller gates passed 1,662 and 2,004
  checks respectively, plus 63 nested allocation, 93 atomic parameter-binding
  allocation and 89 owner-allocation checks:
  `/tmp/nanolang-nested-source-runtime.log`.
- Both ordinary local-name producer methods passed in 0.508 seconds:
  `/tmp/nanolang-nested-source-names.log`.

The two new positive fixtures retain mixed nominal records, declaration-reordered
constructor fields and destructure patterns, repeated calls with reversed
caller paths, shared aliases, disjoint exclusive siblings and observable
mutation. Complete canonical dumps agree across C, raw selfhost, Stage1 and
Stage2 producers. Selected-shadow dumps agree too. Named and name-stripped
modules execute under VM and ASan/UBSan/LSan native. The path-depth boundary
accepts 32 nested fields and refuses 33 without overwriting output.

The initial paired gate covered fourteen ordinary refusal cases. Review adds
explicit reuse of a moved parent, reuse after parent destructuring and a live
tree at source scope exit. All seventeen refusal cases passed the focused
paired rerun in 137.060 seconds:
`/tmp/nanolang-nested-source-refusals-r2.log`. The last control exercises canonical ownership
checking only: raw lowering does not claim to perform that check or execute
selected shadows. I do not bypass the source checker to claim implicit tree
disposal acceptance.

I retain an initial authoring build failure in
`/tmp/nanolang-nested-source-bootstrap.log`. A patch script selected the first
field-access branch inside a newly added helper instead of the intended
expression function, deleting intervening helper declarations. I corrected
the patch anchor before the successful fresh bootstrap. This was a compiler
build failure, not a measured runtime defect.
