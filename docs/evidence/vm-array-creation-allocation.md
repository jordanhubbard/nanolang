# My checked array creation acceptance

I qualify taskaeff9 with the one-line OP_ARR_NEW allocation check at `af3b0d1c`,
restacked unchanged as `6f67b29f`. I return VM_ERR_MEMORY before publishing an
array value when my existing heap constructor returns NULL. My constructor
already frees partial allocation storage. No managed opcode admission changes.

I first applied the guard, then ran fresh ordinary boxed-string and unboxed-int
construction controls. Deterministic allocation refusal covers descriptor and
backing-buffer allocation independently. I require unchanged output and exact
allocation/free counters, empty invocation stack/frames, then four successful
create/release cycles in the same VM after each failure. Successful cycles
preserve live allocated-minus-freed bytes and object counts.

My first fixture incorrectly treated cumulative allocated bytes as live bytes.
I preserved that failed executable/log/hash and static explanation in my
[contract](../NANOISA_VM_ARRAY_CREATION.md), recorded the correction before
editing the fixture, and did not replay the failed artifact. Production stayed
unchanged. Corrected fixture commit `d7b87154` passed:

```sh
make test-vm-substring-contract \
  SUBSTRING_TEST_FLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -O1'
make test-nanovm
```

My corrected focused sanitizer log is
`/tmp/nanolang-array-new-focused-corrected.log`. Full VM acceptance reports
274493 passed, zero failed in `/tmp/nanolang-array-new-fullvm.log`.
I restacked onto main `1826a810` (PR686/687), retaining identical VM and fixture
bytes. The integrated full VM gate passed the same 274493 checks, plus its
allocation/recovery companions, in `/tmp/nanolang-array-new-integrated-fullvm.log`.
I consolidated the duplicate roadmap discovery row into this checked child.

I claim this checked allocation boundary only. Managed array frame/call/global
ownership, collection element-shape admission, parent488 and parent51da remain
open. The historical evaluator791a and Darwin sanitizer7ba remain separate.
