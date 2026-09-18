# My multi-parameter source borrow acceptance

I implement MAC `task_f209d694d3be415982456b00b4df5ad9` after PR #606,
from canonical base `4c2ae731`. Contract `65ca3610` precedes implementation
`bb8887d8`; test checkpoint `4648a9f1` adds both alias orders and duplicate
formal refusal. My verifier, ownership wire format and VM/native runtime are
unchanged. The broader affine and borrow parents remain open.

My helper accepts one through eight exact borrowed record formals. I retain
per-position nominal identity and mode, prepare contiguous caller descriptors
in source order under one region, and address helper fields by their actual
parameter descriptor index. Shared aliases remain shared. Any overlapping
pair containing exclusive authority is refused before publication. Nested
paths, value/reference mixtures, control flow, imports and deeper calls remain
outside my source profile.

I measured these gates on that source:

- Fresh default three-stage bootstrap passed:
  `/tmp/nanolang-multi-source-bootstrap.log`.
- All nine paired source-borrow methods passed in 147.917 seconds:
  `/tmp/nanolang-multi-source-paired.log`. I reused the completed bootstrap
  with `make -o bootstrap test-source-borrow-emission`; I did not skip a test.
  This also passed 123 local-name codec checks, five name-allocation boundaries
  and twenty marker-allocation boundaries.
- My existing multi-caller gate passed 2,004 checks, 93 atomic binding/allocation
  checks and 89 owner-allocation checks:
  `/tmp/nanolang-multi-source-runtime.log`.
- Both unchanged ordinary local-name producer methods passed in 0.477 seconds:
  `/tmp/nanolang-multi-source-names.log`.

My new fixtures include three mixed-mode parameters over two distinct nominal
record types, observable exclusive mutation, reversed caller roots on a later
call, eight weighted scalar inputs, and eight shared aliases of one owner.
Complete C/raw-selfhost/Stage1/Stage2 canonical dumps agree. C and selfhost
selected-shadow dumps agree too. Every formal keeps its correct advisory slot
and zero beginning offset. VM and ASan/UBSan/LSan native executions pass,
including existing name-stripped controls and source publication checks.

Ten ordinary refusal cases cover arity, mode, exact nominal mismatch,
exclusive/exclusive overlap, both shared/exclusive alias orders, mixed value
formals, nine parameters, duplicate formal names and a failed selected shadow.
All three canonical producers preserve old output; raw emitters refuse the
nine structural cases without overwriting output. Raw emission does not claim
to execute selected shadows. Existing one-parameter source gates remain intact.
