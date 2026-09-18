# My managed decimal conversion contract

I record child `task_34ce900cad86496b876bdf262b46bdaf` under my open managed
runtime parent `task_51da49b39230468784da3481b893563b`, after merged substring
PR644. This is my next portable conversion dependency; it does not close my
required floating parser/formatter or general CAST_STRING work.

My VM's CAST_INT string branch calls strtoll with base 10, ignores the end pointer
and errno, then releases its operand. My closed executable begins in the C locale
and does not admit locale-changing imports. I therefore match the C-locale byte
rules: skip space, tab, newline, carriage return, vertical tab and form feed;
consume one optional sign and decimal digits; stop at the first other byte or
embedded NUL. No digits yields zero. Overflow saturates to INT64_MIN/MAX.
I do not claim matching an embedding host that externally changes locale; such
host configuration belongs to the separate host/import contract.

I implement bounded unsigned accumulation against the appropriate signed limit,
without signed overflow, and preserve the exact negative endpoint. This parser
needs no allocation or libc import. It reads a borrowed immutable handle view;
it neither retains nor releases that handle. Existing emitted consumed-operand
cleanup releases the input exactly once, including dynamic aliases across calls
and globals. Runtime view failures set the first status and follow normal frame
cleanup before exported failure.

I route managed string operands through this helper. Existing non-string
CAST_INT behavior remains unchanged, including checked float conversion. Only
my managed closed profile gains CAST_INT for string-bearing modules. Scalar and
literal-only API decisions remain unchanged. String CAST_FLOAT and CAST_STRING
stay refused until their own matched portable implementations are available.

I require VM/native/import-free Wasm comparisons for ordinary signed decimal
prefixes, six whitespace bytes, empty/no-digit text, embedded NUL, endpoints and
saturation; literal/dynamic aliases, calls, globals and repeated-entry cleanup;
non-string cast controls and existing error cleanup; and separate profile and
output-preservation tests. I test corrected ordinary inputs and do not re-execute
historical failed artifacts. Actual target evidence accompanies implementation.
