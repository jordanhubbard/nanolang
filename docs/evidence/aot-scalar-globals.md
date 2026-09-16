# My scalar global storage

I now emit `LOAD_GLOBAL` and scalar `STORE_GLOBAL` using tagged storage. A
global starts with the void tag, not a fabricated integer zero. Stores preserve
integer, boolean and string tags; copied tagged values preserve void as well.
Loads copy the current value, so later global writes do not change saved scalar
values. My module initializer runs before entry and shares the same storage.

I inspect complete 32-bit global operands before classification and size storage
through the highest referenced slot. My VM and translator share the existing
4,096-slot limit. Tests reject 4,096, 65,536 and UINT32_MAX for both loads and
stores, and exercise slot 4,095. I do not truncate operands to 16 bits.

I extend tagged boolean consumption, truthiness, integer/string casts and
scalar printing. Boolean operations check the tag before using the integer
payload. Tests distinguish true from integer one, return a tagged boolean
through a function boundary, and trap typed consumption of uninitialized
globals. A map-free module exercises startup and cross-function mutation.
Another test retains fetched strings in globals after map deletion and global
overwrite, and checks exact printed output for void, boolean, integer and string.

I preserve strings using the existing entry-lifetime storage. This is not
early reclamation or a general foreign-string lifetime guarantee.

I verified:

- `make -j1 test-nvm2c`: 1,241 AOT and 994 shape checks pass.
- `make -j1 test-nvm2c-sanitizers`: the same checks pass with fresh verified
  ASan/UBSan translator and shape objects; leak checking remains disabled.
- `make -j1 test-verifier`: all 95 tests pass.
- Opcode case parity and all three sanitizer-driver unit tests pass.
- `make -j1 test-one-ir-compiler` progresses beyond the global load but fails
  at function 323, offset 1761: field 1 of parameter 0 of function 274 has
  conflicting ordinary-string and tagged-value representations.

I have not completed globals. I still require aggregate storage, empty-array
element inference and full compiler acceptance. I also need compatible record
argument field widening at the newly exposed call boundary. The global task
`task_bcc4271b0de244c2810c09e004f6cd2e` remains open; MAC again refused my claim
with `agent_status_unavailable`.
