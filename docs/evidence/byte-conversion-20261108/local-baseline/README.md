# My byte conversion baseline

I preserve the unchanged complete byte-conversion fixture and three explicit source methods before implementation under #1110. My older local providers pass the whole fixture, whose main returns zero; its conversions execute during selected-shadow creation, not in the resulting native main. Two new production controls pass source creation, verification and VM execution, then refuse the exact missing `vm_bytes_from_string` native ABI. I do not claim generated native conversion execution.

The private ZIP contains 43 files, including three NVM modules, the whole-fixture native program, exact diagnostics/commands, source and observed provider hashes. Its CRC and every file hash independently verify. Deterministic gzip preserves all 39 text files. Failed-control provider hashes were observed after execution; I do not invent before/after custody for those attempts.

I retain my initial fixture-syntax error separately: `array<u8>=` lexes as a greater-or-equal token. Adding whitespace before the assignment corrects the two test fixtures; it does not change my parser. My corrected baseline still retains both real missing-ABI failures.

`docs/SOURCE_BYTE_CONVERSION_CONTRACT.md` records C/interpreter/VM implementation differences. Invalid UTF-8, embedded NUL and size behavior still require complete-boundary observation before choosing native semantics. Fresh source, actual native checker, original prerequisite/translator gates, generated lifetime and platform/release acceptance remain open.
