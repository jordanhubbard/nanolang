# My explicit WebSocket public-provider checkpoint

I build on `aed9d3fb4`. I implement copied host grants, independent connection
and lookup authority, revocation, shared process-local serialization, public VM
execution, public native emission, and an installed C99 archive/header closure.
I do not yet enable paired compiler source admission.

My installed consumer exercises three complete checked bytecode programs with
connection permission denied, and the hostname program with connections allowed
but lookup denied. VM and generated native return the rights error without
opening a socket. Standalone installed consumers exercise both VM and emitter;
native consumers link neither entrypoint. Generated native checks null/revoked
grants, preserved outputs and BUSY refusal with poisoned argument pointers.

My instrumented grant test checks immutable copied policy/path, invalid revision,
deadline and resolver paths, revocation/destruction, output preservation, and
concurrent/reentrant BUSY before argument access. LLVM provider builds use
ASan/UBSan with leak detection; shared ordinary query objects and the installed
archive are not sanitizer-instrumented. GCC separately builds providers and
standalone consumers. I retain exact commands/results in the adjacent JSON.

My shared File indirect-dispatch regression passes both methods in 99.947 seconds.

My complete public real-peer suite **does not pass in this environment**:
`socket.bind(("127.0.0.1", 0))` fails with `PermissionError: Operation not permitted`
before traffic or native parity runs. I retain that original failure. I do not
skip it or treat the installed denial tests as substitutes. The full target
`make -f Makefile.gnu CC=<qualified compiler> test-websocket-public` must pass on
an allowed host, followed by required Linux/Darwin qualification.

An initial standalone sanitizer probe reused hooked socket objects without their
fixture symbols and failed to link. I add explicit no-I/O assertions in the
grant fixture and retain the corrected build/run in the final test receipts.

Paired C-seed/Nano source lowering, public real-peer/fault/cleanup qualification,
public DNS integration and final release gates remain open under #990/#982.
