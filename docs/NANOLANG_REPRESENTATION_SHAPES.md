# My representation-to-shape adapters

I compose the twelve logical C graph adapters under #1142 before whole-module
classification. My Graph solver, storage kinds, carrier masks and fact joins
remain separate qualified dependencies. I grant no image, descriptor, table,
payload, ownership or executable admission through an adapter result.

My current mutable State holds the graph, tracking/generic-array/private-map
selection, source function/offset/opcode and independent first buffer failure.
A Result carries updated State and a C-compatible integer return. Graph mutation
still happens when a prior buffer failure is set if the C helper performs it.
A tracking-disabled helper returns its exact bypass result. Direct graph errors
are wrapped only when shape_ok executes; short-circuit return can leave a graph
error without a buffer diagnostic. I preserve that boundary.

I preserve every physical uint8 kind and uint16 carrier mask, exact shape-kind
versus private-storage namespaces, optional/finite/numeric payloads, inferred
scalar field conversion, record-return field order and present boxed payload
obligations. Unknown array elements do not become integer storage during
resolved reads. Generic arrays intentionally omit exact element constraints.
Logical source contexts and IDs are uint32. Nonphysical inputs return invalid
without mutating the graph; callers retain module/table bounds.

Private-map presence selects the original STRING-key/exact INT-or-STRING value
shape only. It is an input fact, not a validated opaque plan. The actual copied
map field-descriptor accessors and their module-image membership remain a later
composition requirement. I do not port those authority APIs as booleans.
Operand-stack push/pop/origin updates likewise remain classifier work.

I require actual unchanged C graph/helper execution against ordinary/sanitized
references and all five NanoLang routes, complete branch-domain matrices,
recursive/shared graph and poison/disabled/generic/failure controls, meaningful
shadows and original limits/identity guards. Implementation and component
qualification remain pending; full module/body/runtime/platform/release gates
remain required.
