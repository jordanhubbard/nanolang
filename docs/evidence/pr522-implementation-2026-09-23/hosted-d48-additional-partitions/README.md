# Additional d48 hosted failures

I retained completed failures from run `35974675846`, units 08 through 11. These jobs ran historical source `d48a78783`; they are diagnostic evidence, not final-candidate qualification. The snapshot still has later jobs running.

- Units 08: ordinary-record authority links instrumented objects without sanitizer runtimes.
- Units 09: union ownership transport does not return the rejection expected by its existing C assertion.
- Units 10: canonical nested-record-array output attempts to execute a missing `bin/nanoisa_emit`.
- Units 11: the literal PCH-marker negative test unexpectedly succeeds.

I retained each terminal failure section, filed the four tasks in `tasks.json`, and added roadmap entries. Causes beyond these observed failures and final-source qualification remain open; I have not weakened their assertions or classified them as infrastructure failures.

The repetitive linker failure is retained losslessly as `job-107552438029-failure.log.gz`.
