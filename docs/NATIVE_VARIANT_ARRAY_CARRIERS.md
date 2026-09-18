# My finite integer-array variant payloads

I extend task `task_7eb14a92a31d4c6982292f93165f65e5` from the scalar
carrier contract. Shared declaration IDs can put scalar and `array<int>`
instances in the same variant field. I admit that finite set explicitly;
I do not infer permission from an arbitrary tagged value.

Before implementation, my contract is:

- I retain a distinct payload shape for int, bool, float, string, and exact
  `array<int>` members. Scalar-only storage retains its existing shape.
  Conversions into the larger set are directed; exact unification never
  equates arrays with scalars. Array producers retain their exact element
  constraints. Unresolved present producers fail after constraint solving.
- I box constructor array handles with both the array tag and integer element
  storage kind. Copies preserve handle identity. Existing root traversal owns
  reclamation; I neither copy array contents nor change collection scheduling.
- I preserve field provenance through record parameters, locals, returns,
  variant joins, and padding. Missing payload slots remain distinct from
  unresolved present values. Scalar consumers and integer-array extraction
  keep runtime tag, storage-kind, and presence checks.
- I preserve borrowed/owned aliases across calls and collection. I test empty
  arrays, mutation, returned handles, string arms, and repeated allocation.
- I retain explicit refusal for other array element types mixed into this
  finite carrier, records/maps, arbitrary unknown values, and unrelated
  struct/tuple widening. General heap unions remain separate work.

I require same-module VM/native ordinary cases in both producer orders,
source generic instances where supported, GCC/Clang sanitizer ownership
checks, directed solver controls, existing scalar-carrier regressions, and
previous-output preservation. I retain first failed gate logs. I do not
execute historical compiler failures. Full reconstruction remains open.
