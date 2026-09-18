# My absent union payload fields

My native classifier keeps fixed-width field vectors. `AGG_PACK` declares the
actual payload width. Slots beyond that width contain no source value and
provide no scalar type evidence. I leave those unused slots unknown for union
variants; I retain every actual field's exact kind and recursive shape facts.

I do not widen conflicting present fields, change variant identities, weaken
field-access checks, or infer a runtime tag from padding. An empty variant can
flow beside a variant with scalar fields without inventing integer payloads.
I test normal constructors, parameter and return transport, both caller orders,
and unchanged VM/native results; existing present-field conflicts still refuse.

MAC: `task_a4b730306b84428da1f4e5683697353f`. The retained ordinary source is
`/tmp/nanolang-union-lexical-positive.nano`; parent source acceptance remains
`task_9a12d05fafaf4f92bea5919683f9ba32`.
