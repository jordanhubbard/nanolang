# My ordinary records alongside resource owners

I track this required 5.1 correction in [#987](https://github.com/jordanhubbard/nanolang/issues/987), under #976. My existing nine-method affine module-identity suite is the unchanged source contract. I retain [baseline and isolated candidate evidence](evidence/affine-product-regression-20261009/README.md).

## My observed boundary

At 5dc3a17fb, seven component subcases stop in self-hosted lowering. My isolated emitter candidate preserves ordinary alias loads, emits ordinary aggregate construction, routes ordinary argument/result values without owner moves, retains nested ordinary field identity, and accepts qualified owner result calls. Both candidate source builds pass existing compiler shadows; the extended build adds a useful ordinary-receiver shadow.

The extended unchanged suite still has seven failures, now all in module verification. Source admission is insufficient. I have not integrated or qualified this candidate, and I do not bypass verification.

## My shared contract changes

I must derive ownership from the complete layout's RESOURCE flag. Currently `nvm_affine_type_is_owned` treats every complete STRUCT as owned; it must continue to require valid complete metadata, while distinguishing copyable records from resource-bearing records. Incomplete or malformed layout facts cannot authorize copying.

I need exact ordinary-record dataflow for construction, load/store, projection, parameters and results. The bytecode analyzer currently rejects non-union AGG_PACK, labels every STRUCT load an observation, and lacks an ordinary-record STORE_LOCAL route. I must carry layout identity across stack joins, calls and returns for copyable records as well as owners. Different layouts with the same tag cannot meet or substitute for one another. Local joins must preserve definite initialization for ordinary records.

I must validate all constructor field tags and nested identities, reject any owner or borrowed observation embedded in an ordinary record, and retain ownership checks for OWN_PACK/OWN_MOVE_LOCAL/OWN_STORE_LOCAL. Ordinary field projection must preserve the child layout and may not project an owner as a copy. General declaration ordering, managed payloads and resource nesting remain part of the full release requirement; a scalar fixture is not full admission.

I must extend exact parameter/result queries and call-graph validation without granting borrowed references escape authority. Native and VM dispatch consume these queries, so I must audit both runtimes before publishing the new verified route. The private mixed-array proof is a separate authority path; I do not relabel this module to bypass the ordinary verifier.

## My required checks

I require raw malformed-module controls for owner copying, incomplete metadata, wrong constructor fields, mismatched record arguments/results, cross-layout stack joins, uninitialized local joins, observation escape and resource-bearing children. Positive raw controls must execute construction, aliasing, nested projection and calls/results through verified VM and sanitizer-native products.

I then require the unchanged nine-method source suite through C seed and fresh Stage1/Stage2, adjacent ownership/borrow/mixed-array tests, full native checks, fresh raw Stage1/Stage2 equality and Linux/Darwin qualification. Existing resource failures must preserve prior output. I retain first failures and do not reinterpret refusal as success.
