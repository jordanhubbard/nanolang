# LinkedIn post — NanoLang 4.5

Paste the block below. Lead with the sandbox. Services exist so a
program can touch the host without becoming it. Forth is an ISA proof,
not the product. The technical boundary remains `docs/RELEASE_4.5.md`.

---

NanoLang 4.5 is public.

The problem is not that machines write code. The problem is that the
code they write wants the host: files, the network, other processes,
GPUs, Python. Most runtimes hand the whole machine to whatever runs. I
do not.

I am a language meant for programs that machines write and people can
still read. This release is the runtime that sits between that program
and the operating system. You say what it is allowed to do. I refuse
the rest.

That is what a service is for.

A useful program has to ask the host for something. A service is that
ask, made into a door: who may knock, what they send, what they get
back, and what of the machine that door may touch. You describe the
door once. I generate the clients in NanoLang, Python, Rust, and C++.
An older client can ask whether a newer door is still safe. If the
process behind the door dies, your program does not have to die with
it. Generated code can do real work — log, read a file, talk to a
neighbor — without becoming the operating system. That is the point.
Without services, the only honest answers are “no host at all” or
“here is libc, good luck.”

The security model is the rest of the story.

Permissions are tokens I mint from host entropy. You cannot forge one
from an integer or a pointer. You can hand someone a weaker grant; you
cannot keep the original after you transfer it. Restart invalidates
old tokens. Declared effects become a reviewable deployment manifest:
uncovered grants fail closed, unused grants are counted, an override
does not rewrite the source. A local capability does not leave the
machine. FFI already ran in a co-process; the editor’s language
process now does too. If it crashes, the window stays up. I do not
claim GNU Emacs. I do not claim a kernel. POSIX is the host. I isolate
on top of it.

When a run is nondeterministic — time, entropy, the network, user
input — a journal can record what happened and play that answer back
without calling the original service. In this tag the journal is a
library you call, not a hook on every trap.

I proved the bytecode path with Forth because Forth is a language, an
assembler, and a compiler in one design: a single stress test of the
instruction set, not a product direction and not a return to 1970. I
record which suites pass. I do not call myself a Forth Standard
System.

What did not change: bytecode is verified, not merely well-formed.
Every function still ships with the test that describes it.

Release, deck, and narrative:
https://github.com/jordanhubbard/nanolang/releases/tag/v4.5.0
https://docs.google.com/presentation/d/1oWP5WJ7q5XhUF5jB_iLf3qO1mTdtrNt3FqIvYfbH2uM/preview
https://docs.google.com/document/d/1AHbhUecsOx2QHG4fTMlFDA7l4xZR9IhhgV80NmdiCb8/preview
