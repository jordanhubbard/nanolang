# My character operand lifetime evidence

I implemented `task_62caf894db9649cd904ce6faf3c37ffb` at frozen source
`e2418349`, after contract `10f7787f`. STR_CHAR_AT now releases the popped
index after its last use on success, and before source-type refusal. I retain
integer unsigned-byte/-1 results and the existing non-integer fallback to zero.

My fresh corrected-source lifecycle fixture exercises integer indices -1,
0, 1, 2, 3 and INT64_MAX over `a`, embedded NUL and byte255. Repeated calls with
an ordinary retained string index preserve the caller's single owner while
using fallback zero. Source-type refusal clears transient stack/frame state
and preserves the caller alias. Final release restores the heap baseline.

I passed `make test-vm-substring-contract`, then the same target with
`SUBSTRING_TEST_FLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -O1'`.
This instruments the compiled VM/heap/value/cycle components of that focused
harness; it is not an all-library sanitizer claim. I then passed full
`make test-nanovm`: 274493 checks, zero failures, plus the required allocation,
callback and stack recovery companions. Parent independent production review
found no scoped blocker.

Logs remain `/tmp/nanolang-char-lifetime-build.log`,
`/tmp/nanolang-char-lifetime-sanitized.log` and
`/tmp/nanolang-char-lifetime-full.log`. I did not execute a historical failed
artifact or reproduce a pre-fix failure. This is a defensive ownership repair,
not LLVM/Wasm admission or closure of runtime51da, Darwin7ba or evaluator791a.
