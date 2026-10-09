# File shadow suite supervision

I add a byte-only supervisor under #989 from parent `79d325f6f` on Darwin.
My independent frontends remain responsible for selecting and lowering every
required shadow. This component accepts their emitted modules and origin/name
labels; it does not accept ASTs or delegate frontend semantics.

I use one child process and one ten-second default deadline for the complete
suite. The bounded `NANO_SHADOW_TIMEOUT_SECONDS` override is shared with my
existing drivers. Each invocation creates, revokes and destroys its own explicit
temporary-file grant. I require successful runtime acquisition, clean cleanup and
a zero scalar result before recording completion. I never persist a grant.

I create an exclusive mode-0600 log in the caller's private staging directory.
I hex-encode names and origins, record the entire selection before execution,
and flush/fsync each start/completion record. I require a separate complete
child report plus successful child exit. I kill the child process group on
success or failure and bound direct-child cleanup after timeout. This is not a
security sandbox. On timeout or absent final report, the returned completion
count is zero; durable log records retain any completed prefix.

## Evidence

- `final-controls.log`: strict C99 build and supervisor fault controls pass.
  I cover explicit denial before log creation, selected/start/done equality,
  exclusive path and symlink refusal, assertion failure, two 600-ms invocations
  exceeding one whole-suite one-second deadline, missing completion despite
  successful exit, signal death, runtime cleanup failure, missing acquisition,
  nonzero result, grant creation/revocation/destruction failures and termination
  of a descendant that would otherwise write a marker after its parent dies.
- `final-sanitized.log`: the same supervisor/fault harness passes with LLVM
  AddressSanitizer and UndefinedBehaviorSanitizer at `-O1`. The harness substitutes
  File callbacks to isolate supervisor failures; this is not instrumentation of
  the entire File runtime. Forced child `_exit` paths do not run exit-time leak
  checks.
- `final-source.log`: all nine C source methods pass in 50.644 seconds at the
  production native `-O1` setting. I execute the five unchanged generated shadows
  separately and as one real supervised suite, compare the full selected/start/
  completion records, and refuse an actual failing shadow with a live File.
  Existing VM/native source behavior, grants, ownership and cleanup controls stay
  enabled. Generated native consumers use ASan/UBSan; the common compiler objects
  and File archive are ordinary builds.
- `nano-source.log`: my added failing-shadow source passes through both compiler
  producers and VM/native executions of the independent Nano lowerer (one method,
  123.983 seconds including probe builds). Emitted bytes agree across all four
  routes and actual consumer assertion/cleanup results agree. The second producer
  is my retained C-produced development driver, not a release fixed point.
- `source.log`: first nine-method run passes in 51.822 seconds before I add the
  complete five-shadow suite check.
- `build.log`: the first strict Darwin build fails because POSIX feature macros
  hide `mkdtemp` and `O_NOFOLLOW`. I declare the Darwin feature set explicitly;
  `final-build.log` records the corrected build and controls.
- `inputs.json` pins the implementation and test source hashes.

## Remaining integration

I have not yet connected this component to the actual C/Nano compiler CLI
publication paths. Root/import shadow selection, the Nano byte-only host bridge,
compiler/runtime grant flags, provisional output staging and atomic publication
remain required. These component tests do not qualify installed Linux behavior,
a release bootstrap, all backend/profile work or the full 5.1 scope.
