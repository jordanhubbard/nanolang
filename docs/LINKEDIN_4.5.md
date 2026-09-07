# LinkedIn post — NanoLang 4.5

I am NanoLang. I tagged `v4.5.0`. The last public GitHub Release was
`v4.0.0`. The five product phases in between shipped as this one tag.

I am a language designed for machines to write and humans to read. I
also host a secure runtime: versioned service contracts, unforgeable
capabilities, a POSIX supervisor, and a trap journal. I run on an
ordinary kernel. I do not claim a kernel of my own.

Since 4.0 I added a Forth session that compiles colon definitions to
verified NanoISA. Jackson Core and Core Ext suites are vendored and I
record what they pass. I do not claim a Forth Standard System.
`INCLUDED` is still a gap.

I added UTF-8 message catalogs for six languages and machine-draft user
guides. Human stderr can follow the process locale. JSON and TOON stay
English. I do not call the system internationalized.

I added Nano Service Interface v0: stable ids, fail-closed documents,
generated stubs, and invocation by method id. On top of that I added
unforgeable capabilities and a POSIX service fabric. The SDL editor’s
walker now runs in `bin/nano_emacs_worker`. The frame does not load the
interpreter in-process. I do not claim GNU Emacs.

4.5 maps declared effects to least-privilege grants, records
nondeterminism in a journal, and copies trace ids across NanoVM,
router, service, and host. Replay returns the recorded result. The
journal is a tested library in this tag, not a hook on every VM trap.

4.0’s contract is unchanged: bytecode is verified, not merely
well-formed. Shadow tests still ship with the function they describe.

`docs/RELEASE_4.5.md` is the boundary. GitHub:
https://github.com/jordanhubbard/nanolang/releases/tag/v4.5.0
