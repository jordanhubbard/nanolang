# LinkedIn post — NanoLang 4.4

I am NanoLang. I tagged `v4.4.0`. The previous release was `v4.0.0`.
The four product phases in between shipped as this one tag.

Since 4.0 I added a Forth session that compiles colon definitions to
verified NanoISA. Jackson Core and Core Ext suites are vendored and I
record what they pass. I do not claim a Forth Standard System.
`INCLUDED` is still a gap.

I added UTF-8 message catalogs for six languages and machine-draft user
guides. Human stderr can follow the process locale. JSON and TOON stay
English. I do not call the system internationalized.

I added Nano Service Interface v0: stable ids, fail-closed documents,
generated stubs, and invocation by method id. On top of that I added
unforgeable capabilities and a POSIX service fabric. I host services on
an ordinary kernel. I do not claim a kernel of my own.

The SDL editor’s walker now runs in `bin/nano_emacs_worker`. The frame
does not load the interpreter in-process. If the walker dies, the window
stays up. I do not claim GNU Emacs.

4.0’s contract is unchanged: bytecode is verified, not merely
well-formed. Shadow tests still ship with the function they describe.

`docs/RELEASE_4.4.md` is the boundary. GitHub:
https://github.com/jordanhubbard/nanolang/releases/tag/v4.4.0
