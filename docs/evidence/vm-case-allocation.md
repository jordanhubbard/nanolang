# My checked VM case-conversion evidence

I implemented `task_543fe0e46aa34404b5cae96935267d10` at frozen source
`18f23ae3`, after pre-code contract `feacacfb`. I check scratch-size arithmetic
and final string allocation before publication, preserving ASCII-only byte
conversion and existing input/scratch release. No LLVM/Wasm admission changes.

My first fresh fixture incorrectly required an allocation failure for an
empty result. Its ordinary assertion stopped before acceptance. Static review
of `vm_string_new` established the reason: an interned equal string is returned
before allocation. I recorded the correction in roadmap commit `cf80be98`
before continuing. I preserved `/tmp/nanolang-case-allocation-focused.log`
and `/tmp/nanolang-case-allocation-original-test`, SHA256
`3fac56132f5eb8b8dd181eb598282799215f3fdf0bd54eac68eecab6dd996112`;
I did not rerun that artifact. This is a demonstrated test-contract mistake,
not an unexplained infrastructure failure or an executable misuse test.

My corrected fixture covers lengths0/7/255/256/300 in both modes, including
NUL/high bytes and both letter cases. It requires successful interned empty
output with allocation disabled, checked MEMORY status for changed bytes,
unchanged caller aliases, subsequent successful byte conversion, and complete
release. I also corrected the managed trim contract: its private runtime
allocates fresh results, while VM interning may reuse objects. Matched bytes
and ownership do not imply identical allocation events or physical handles.

I passed `make test-vm-substring-contract`, then focused ASan/UBSan with
`SUBSTRING_TEST_FLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -O1'`.
Those flags instrument the compiled VM/heap/value/cycle components of the
harness, not every linked library. Full `make test-nanovm` then passed all
274493 checks plus allocation/callback/stack companions. Logs:
`/tmp/nanolang-case-allocation-corrected.log`,
`/tmp/nanolang-case-allocation-sanitized.log`,
`/tmp/nanolang-case-allocation-full.log`.

I leave managed case conversion, full runtime51da, Darwin7ba and evaluator791a
open. Corrected ordinary acceptance does not close those separate obligations.
