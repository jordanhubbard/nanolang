# My checked native union projection

My retained Linux4d native-nested-list-projections case passes checking but fails
host C compilation. Its payload alias is emitted as nl_Outer.Wrapped, although
its retained checker view owns Outer<Item> and the exact selected Wrapped arm.
Its direct constructor match is emitted as nl_Outer despite constructing
nl_Outer_Item. I do not execute either rejected product. Both losses are emission
metadata defects; they do not justify admitting a new nominal identity.

I keep the original source and all assertions. I project native storage from a
complete NominalView, resolving its declaration through owner-aware identity and
substitution before materializing its arguments. A selected payload retains the
same union declaration and a checked variant ordinal. I copy the concrete union
annotation and selected variant name into independent output owners before
publishing either. Failure leaves caller outputs unchanged and frees only new
copies. I do not infer the variant by splitting a dotted materialized name.
This emission projection is not a replacement for checker authority.

For inferred payload declarations, I use the initializer's actual retained view
before publishing the new lexical binding. The native declaration combines the
same concrete union key used by layout emission with its checked variant name.
Ordinary records continue through their existing declaration path. Missing or
inconsistent required payload proof refuses lowering; it cannot fall back to
nl_Outer.Wrapped. The previous same-name lexical binding remains the initializer's
source. No Symbol pointer survives a recursive helper that can grow its storage.

Both expression and statement match checkers must consume complete scrutinee
facts, including direct constructors, aliases, calls, and nested fields. I replace
the partial AST-shape inference with the checked projection for union scrutinees,
retaining original integer/domain and arm checks. Their owned concrete annotation
and owner remain live through arm checking; recursive bindings cannot invalidate
them. Native match metadata records the concrete key only after successful
projection, and failed checking retains existing rollback obligations. No
ownerless spelling lookup becomes authority for binding the selected payload.

Before implementation I audit ordinary and generic unions, exact imported owner
keys, both match paths, payload alias declarations, and generated layout naming.
Additive controls exercise these actual consumers with local/imported fixed and
substituted leaves, nested payload aliases, direct constructor/call/field
scrutinees, and same-name binding growth. Checked allocation-prefix controls
preserve sentinel outputs and prior proof on failure, then recover independently.
Original18 plus native9 remain required after source review. Nano producer parity
and all earlier failures remain separately attributed; this design is unexecuted.

MAC: task_a370d7cc60c5445981c0fbc5d7b15f29.
