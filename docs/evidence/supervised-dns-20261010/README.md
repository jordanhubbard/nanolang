# Supervised private DNS

I implement `nl_socket_resolve_tcp_supervised` under issue #990. My existing
synchronous adapter validates counted names and caller storage first with lookup
disabled. Numeric addresses return directly; denied DNS lookup starts no process.
For permitted DNS, my caller supplies an absolute trusted resolver executable
and a 1..60000 millisecond deadline. I use `posix_spawn` without a shell or PATH
search; I do not call the resolver in a post-fork copy of the caller.

My standalone `nano-resolver` copies the existing resolver's deduplicated IPv4/
IPv6 endpoints into a fixed-size, versioned big-endian packet. My parent accepts
exactly one complete packet, EOF and a successful child exit before publication.
I validate status/error domains, count, address families, requested port, IPv4
padding/scope, duplicates and unused packet bytes. I preserve prior caller output
on failures and keep resolver errors separate from supervisor errors.

My parent reads nonblocking output against a monotonic deadline. A timeout kills
the worker process group and reaps the child; a full response from a still-running
worker is insufficient. I retain the child unreaped until EOF so a process-group
identifier cannot be recycled while a descendant holds the pipe. Process cleanup
still relies on operating-system progress. My serialized caller must not reap this
worker from another thread or signal handler. This is not a sandbox: the trusted
helper inherits host authority and ordinary non-close-on-exec descriptors.

## Tested on Darwin

`qualification.log` contains the exact final Clang, GCC 16 and LLVM
ASan/UBSan/leak-check build and execution commands. All three runs pass:

- Real localhost resolution with copied endpoints and the requested port.
- Numeric resolution and lookup denial without a helper.
- Eleven malformed-worker responses: short/extra bytes, wrong version, family,
  port, count, unused tail, errno domain, duplicate endpoint, status and
  inconsistent resolver status/error.
- Three 80 ms deadline cases: no response, full response without EOF, and full
  response plus EOF without process exit. Each returns within the test's 2 s
  bound, preserves output and leaves no waitable child.
- Nonzero exit before or after a response, signal termination, missing executable,
  relative helper path and invalid deadline refusal.
- Resolver `EAI_SYSTEM` and its original errno remain distinct from supervision.
- Closing each standard descriptor before invocation does not break pipe setup.
- Existing owned socket endpoints still transfer data after refused/timed-out
  resolution and are disposed normally.

`make-and-network.log` records the initial Make target plus unchanged synchronous
hostname/buffer and real IPv4/IPv6 network adjacency tests. Subsequent changes
add packet error-domain validation and cleanup-error reporting; the final direct
qualification compiles those changes. `hashes.json` identifies all source inputs
and the six final helper/test binaries. `verify.sh` reproduces this Darwin run.

## Remaining integration

I build the helper with `make -f Makefile.gnu bin/nano-resolver`; my private host
caller links `src/nsi_socket_resolver.c` alongside the existing Socket adapter.
I have not installed or automatically selected this helper for public service
execution or the WebSocket module. That integration must provide a trusted stable
helper path and carry DNS permission/deadline through a versioned service policy.
String-bearing DNS catalog layouts/results, affine WebSocket bindings and Linux/
exact-candidate qualification remain open. This checkpoint does not close #990.
