# My checked VM replacement evidence

I implement `task_6b3d7306179f4a7786adbd29b9acbe6c` at frozen production
`a6bfd62d`, after pre-code contract `c55aa6d4`. My helper bounds counted removed
spans before multiplication and replacement growth before addition. I check
scratch terminator representability and final/empty-needle allocation before
publication, preserving all three operand owners through cleanup.

Direct scalar helper checks cover shrinking/growing results, exact UINT32_MAX,
excess occurrence count, unrepresentable growth, empty needle and null output.
They construct no oversized strings. Nine fresh ordinary byte fixtures cover
nonoverlapping matches, embedded NUL, deletion, no match, empty input/needle,
interned equal results and a 600-byte source growing to 1200 bytes. Deterministic
final allocation failure preserves caller aliases, clears transient frames and
permits corrected reentry. Existing interned output instead succeeds without
allocation. Repeated successful results preserve exact bytes and reference
counts; source-type refusal also retains the three caller owners.

My initial compilation caught a type-limits warning in a uint32-versus-SIZE_MAX
comparison. I used the equivalent widened `(uint64_t)length + 1 > SIZE_MAX`
guard, whose addition cannot overflow for uint32 length. No executable ran
before that correction. I preserved the initial compile log separately.

I passed the corrected focused lifecycle target and its ASan/UBSan variant:
`SUBSTRING_TEST_FLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -O1'`.
These flags instrument compiled VM/heap/value/cycle sources, not all linked
libraries. Full `make test-nanovm` passed 274493 checks plus required allocation,
callback and stack companions. Independent parent production review found no
scoped blocker. Logs: `/tmp/nanolang-replacement-focused.log` (initial compile),
`/tmp/nanolang-replacement-focused-r2.log`,
`/tmp/nanolang-replacement-sanitized.log`, `/tmp/nanolang-replacement-full.log`.

Managed replacement remains a separate contract. STR_SPLIT requires aggregate
runtime work. Full runtime51da, Darwin7ba and evaluator791a remain open; no
historical failed artifact or pre-fix executable failure was replayed.
