# My module builder can capture its own source

My conservative GCC PCH scan recognizes the bytes `#pragma GCC pch_preprocess`,
even in a C string literal. Both of my scanner implementations contained that
complete literal in their own source. Building that source through my module
builder therefore failed capture before compilation.

I split two C literals into adjacent literals. C concatenation preserves the
runtime marker bytes. Preprocessing no longer mistakes my own spelling for an
external PCH directive. I change neither scanner logic nor publication policy.

`make test-module-builder-self-capture` passes four methods on Linux/GCC:

- My module-builder source passes the normal manifest capture/build path. Its
  object still contains the original runtime marker bytes.
- A complete marker in an ordinary source string still refuses capture.
- A real canonical GCC external PCH is copied byte-for-byte into the retained
  generation and its directive is rewritten to that copy.
- Real GCC PCH output with an unrepresentable quoted path is refused; the input
  snapshot remains unchanged and no temporary rewrite is left behind.

The local compiler-support facade also builds and runs through the normal
NanoLang module path after this change. That facade is a separate pending
compiler artifact-import slice, not part of this repair.

Evidence: `/tmp/nanolang-module-selfcapture-four.log`,
`/tmp/nanolang-compiler-support-probe-trace.log` (before), and
`/tmp/nanolang-compiler-support-probe-after.log` (after).
