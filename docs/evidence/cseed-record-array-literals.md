# My C-seed record-array literals

I construct nonempty record literals in an `ELEM_STRUCT` dynamic array. I
snapshot each child in source order, then copy its value using its actual C
size. This replaces the invalid C compound-array fallback. I reuse my existing
collision-checked argument temporaries and dynamic-array name allocator.

My native/VM fixture checks literal values and returned arrays, nested record
fields, left-to-right child effects, generated-name collisions, record value
copies, and array aliases after mutation. Mixed nominal record types in one
literal reject with E001 and preserve prior output in both frontends.

The focused fixture, direct record projection, nested-array literal tests,
typechecker suite and fresh three-stage bootstrap pass. The final alias/name
collision fixture also passes separately. Logs are
`/tmp/nanolang-record-literal-gates.log` and
`/tmp/nanolang-record-literals-alias.log`.

I track this representation repair as `task_a5fc558cbfa34f14b4d580923de4209c`.
It does not establish complete nominal array assignment checking. A homogeneous
literal of `Second` still passes a declared `array<First>` annotation in the C
frontend; I retain that separate reproduction and full contextual checking task
`task_8c736631e97043729bf465a2d6bdc2d5`. Nested aggregate VM lifetime and native
schema-name collision findings also remain open on my roadmap.
