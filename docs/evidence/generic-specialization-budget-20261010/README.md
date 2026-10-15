# My generic specialization limits

I bound my self-hosted generic worklist without prohibiting finite polymorphic
recursion. I reuse an existing instance before applying allocation limits.
A new instance retains its first-discovery parent. I refuse when creating it
would exceed 64 instances of the same declaration along that ancestor chain,
4,096 unique generic instances in one module, or 1,048,576 bytes of retained
concrete type spellings. Independent root bindings do not consume recursive
depth. I reset all counters and the active instance with my lowering state.

I first added count and byte limits. They were insufficient for prompt refusal
of recursive nested arrays: my first candidate exceeded a 60-second observation
limit. I retain that failure. With the recursive-depth guard, the same program
returns the explicit specialization-budget diagnostic in 0.748 seconds on this
host. This is one measured case, not a general compilation latency guarantee.

My rebuilt C-seed-hosted source drivers and actual `src_nano/nanoisa_emit.nano`
publication driver pass all 13 methods in `final-tests.log`. I qualify direct
and mutual growing-type recursion in whole, selected-program and shadow modes,
finite polymorphic recursion, 70 independent record bindings, and the existing
primitive/record/array generic corpus. Nine positive component cases produce
27 verified modules executed in NanoVM and sanitized standalone C AOT. The
publication driver preserves prior assembly and binary outputs on refusal,
and publishes and executes a verified finite-polymorphic module. My emitter's
mandatory shadows exercise exact count/byte budget boundaries.

These are fresh source-driver results, not rebuilt installed compiler stages.
The full generic and release requirements remain open. My unchanged C producer
still rejects the retained `array-parity.nano` source before shadow publication
(`array-parity-baseline.log`), while the self-hosted suite passes that array
binding case. C aggregate identity parity is the next implementation gap;
structural type-variable inference, contextual callable specialization,
metadata and integrated bootstrap/module-loader qualification also remain.
