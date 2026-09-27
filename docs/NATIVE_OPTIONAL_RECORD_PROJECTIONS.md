# My optional record projections

MAC `task_202627b103cc44b98020a082e2c9deb8` tracks the native failure reached
through the unchanged nested generic record-array acceptance source for #522.
My C bytecode producer and VM already execute that source. My retained native baseline
conflates a tagged optional result with its present record shape.

I retain the optional carrier and constrain its present payload separately at
an aggregate field read. A record read through a tagged array returns VOID for
an absent element, or a tagged plain-record snapshot for a present element.
Consuming that result as a record checks the tag, pointer and aggregate kind
before accessing fields. Existing width and field-storage checks remain.
I do not reinterpret absence or a scalar pointer as a record.

A snapshot copies the record value before the source array can grow or replace
an element. Its nested fields remain live through the existing native root
tracer. Tagged record values add their snapshot to record roots, and normal
sweeping/entry teardown releases snapshot allocations. Read and allocation
failures must preserve the existing native error boundary.

I require the original source through C bytecode and paired VM/native execution,
raw present/absent and wrong-tag controls, copies across array replacement and
collection, allocation-failure checks, ASan/UBSan/leak checks and the complete
translator regression. Self-hosted record/union shape admission remains a
separate unfinished dependency; this runtime repair alone cannot resolve #522.

My native record storage does not preserve VM object identity. I continue to
refuse tagged record comparisons before publication, including EQ/NE and
ordering, rather than compare snapshot addresses or read a record as a string.
