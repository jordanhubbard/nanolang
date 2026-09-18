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

## My implementation and retained evidence

My native implementation checkpoint is `1483b746`, based on canonical main
`8f7fc062`. Constructor fields retain classifier-only member masks; the new
shape is an explicit finite leaf, not an unknown heap shape. My array member
conversion checks its exact integer element, and record joins use separate
destination storage. I preserve strict exact unification. A legacy empty
integer-array placeholder acquires its integer element constraint at this
explicit constructor boundary. Other empty-array contexts are unchanged.

I reuse the boxed array representation `(TAG_ARRAY, integer-array-kind,
handle)`. Existing `nroot_value` follows that kind and handle. My checked
array-result extraction validates the tag, storage kind and non-null handle;
I do not change collector thresholds, owner pools or array allocation.

I retain initial failures separately:

- `/tmp/nanolang-variant-array-build.log` requested a nonexistent Make target;
  the corrected tool build uses `nano_vm`.
- `/tmp/nanolang-variant-array-first.log` identified the missing empty-array
  element constraint. `/tmp/nanolang-variant-array-focused2.log` identified a
  mistaken pointer-to-pointer declaration in my new extraction helper.
- `/tmp/nanolang-variant-array-joins.log` identified exact unification of two
  different finite payload sets at a record branch join. I give that join
  separate destination storage and directed conversions before qualification.

At the corrected implementation, I tested:

- 1,365 solver checks, including directed scalar-set/integer-array admission,
  late element resolution, unchanged producer constraints, exact-unification
  refusal and unknown/non-integer element exclusions.
- 12 methods with GCC (11.831s) and Clang (14.633s), including ASan/UBSan/LSan
  generated programs. I cover both call orders, same/different variant tags,
  empty variants/arrays, record joins, projected calls, checked returned array
  handles, mutations observed through aliases after 2,048 allocation cycles,
  scalar/string regressions and retained-output refusals.
- All 16 unchanged generic-emission methods (58.418s), including their existing
  affine and scalar-match fixtures, using my translator as an external tool.

My new `generic_scalar_array.nano` source passes Cseed, Stage1 and Stage2
emission, VM verification/execution and sanitizer native execution. I use
immutable tools in `/home/jkh/Src/nanolang-canonical-generic-unions/bin`:

| Producer | SHA-256 |
| --- | --- |
| nano_virt | bbfe909261b2b722f5e3f6feddd298922d6684d615992fd5f87bdf35e100e90f |
| nanoc_stage1 | bbd3f8ed7e03427c703c1376c3adafeeaaa0bd894ff6b1d7b97e6195c7288ba1 |
| nanoc_stage2 | 2cbb2c590811aeb92f97b89c38f931e630b39c63a4b59a377c802e352f4f7c77 |

The containing checkout is `afd8e61bee7f9be82bc846b8a01c5cebcc3afb0e`;
I identify the actual binaries by their hashes rather than claiming a fresh
bootstrap from that checkout. My translator hash is
`e1e9d1b6b505a13d40f793c41ce36cf73b7afc6638589da507328f9fbd274170`.
The tool manifest and verification are retained at
`/tmp/nanolang-variant-array-tools.sha256` and
`/tmp/nanolang-variant-array-tools-final-verified.log`. Focused logs are
`/tmp/nanolang-variant-array-{gcc-final,clang-final,generic-existing}.log`.

This is finite native payload acceptance. I do not claim a current-main
bootstrap, a new compiler fixed point, or general heap/array-kind unions.

I also passed the complete 2,422-check native gate with zero failures at this implementation checkpoint (`/tmp/nanolang-variant-array-full-native.log`). The final tool-hash verification passes after all gates. I reconcile the task only after reviewed canonical merge.
