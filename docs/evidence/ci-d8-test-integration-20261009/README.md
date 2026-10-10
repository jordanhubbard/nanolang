# I repair the next CI integration terminals

Under #982 I inspect completed CI38015642804 at `d8c4c495c`.
All three build platforms and coverage fail the isolated Make-link fixture:
its synthetic tree omits `src/service_driver.c`. Sanitizers fail schema coverage
because `FILE_CALL_REFS` is absent from the exact generic-VM exclusion set.
I retain the [CI terminals](ci-terminals.log).

I explicitly exclude `SERVICE_DRIVER_OBJECTS` in the existing synthetic fixture,
as it already excludes all other real provider groups. Its assertions still
check stable unchanged tools, relinking from a changed common object and each
individually missing tool. I add byte0x97 to the schema test's exact private File
set. This matches the existing private execution boundary; I add no generic VM
handler and remove no coverage assertion.

On Darwin, schema generation/current-artifact checks and all33 schema tests
pass (1.275s). All five bootstrap component tests pass (50.599s), followed by
all21 source/receipt/boundary/tool tests (19.350s). The combined command then
stops before File opcode execution because LLVM `opt` is missing from PATH.
I retain [that complete log](local-schema-bootstrap.log).

With `/opt/homebrew/opt/llvm/bin` prepended to PATH, I run only the unreached
`test-file-opcodes` gate. Both methods pass in2.885s, each reporting1,632
encoding/transport/refusal checks, including the existing generic-consumer
refusals for `FILE_CALL_REFS`. I retain [the result](opcodes.log).

Both Make invocations select Homebrew LLVM Clang and
`OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5`. Full exact-revision CI and
the remaining 5.1 implementation remain open.
