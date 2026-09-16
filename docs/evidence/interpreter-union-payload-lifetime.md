# My interpreter union payload lifetime

On 2026-09-16, Darwin CI job `104888740140` failed the shadow for
`scripts/userguide_snippets_check.nano:parse_snippets`. I reproduced the failure
on puck in an isolated checkout of `7a6ac4e8`. The fixture wrote valid JSON and
read it successfully. Its helper held `"x"` before and after releasing the JSON
object, but `Result.Ok.value` was empty after returning to the caller.

My `create_union` constructor copied payload values shallowly. The interpreter
then released the constructing function's local string, leaving the returned
union with a dangling pointer. Linux allocator behavior could leave the old
bytes readable; passing there was not evidence of correct ownership.

I now copy string payloads and use my existing record-copy policy for struct
payloads. Other referenced values retain their existing sharing policy. Copying
a `Value` still shares its union identity; this change adds no union destructor
or independent free operation for aliases. It does not establish complete
interpreter aggregate reclamation or cycle ownership.

My JSON helper shadow now checks the returned string instead of only its Ok tag.
The snippet shadow checks its parsed count, name and source text. A typed local
retains the generic array type before matching the returned result. I did not
skip the failing shadow, narrow the guide selection or lengthen its deadline.

## Evidence

- A focused fresh ASan/UBSan constructor harness releases the original local,
  reads the payload through a copied union `Value`, and releases the owned
  payload/container once. It passes with `ASAN_OPTIONS=detect_leaks=1`.
- The identical harness against the prior `env.c` reports heap-use-after-free
  in `strcmp`, reading the returned payload after the original local's release.
- On Darwin arm64, the patched C seed compiles the checker with the stronger
  shadows; the checker passes all **39 guide snippets**.
- `make test-env-scoping` passes **39 C checks** and **10 lexical-scope methods**
  on Darwin after the repair.

I used the production `env.c`, `gc.c`, `dyn_array.c` and `gc_struct.c` in the
sanitizer harness, with function sections to retain the focused constructor
path. This is bounded lifetime evidence, not a complete sanitized interpreter
run or final release-tree acceptance. MAC: `task_3f227591ba94400a831005e1eec376f5`.
