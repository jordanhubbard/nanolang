# Tagged native map lookups

I emit `HM_GET` as a tagged value, not an integer or string with an invented
default. A missing key remains void. My optional shape keeps the present-value
shape on a separate edge, so a lookup does not unify absence with an ordinary
scalar. Tagged temporaries flow through duplication, swapping, discarding,
locals, same-representation branch joins and compatible function arguments.

I copy fetched strings into entry-owned storage. Replacing or deleting a map
entry does not invalidate an earlier lookup. I release these copies after the
entry returns. This is lifetime-safe for the implemented paths, but it retains
unreachable lookup strings during execution. Early reclamation remains part of
`task_2f837947d5f24130b401ae433dd8d8c9`; entry cleanup is not bounded retention.

Integer and string consumers check the tag before accessing the payload. An
explicit `CAST_INT` or `CAST_STRING` follows my VM's conversion rule, including
zero or empty string for void. `TYPE_CHECK` can inspect a tagged lookup without
consuming it as a scalar. Conditional branches and assertions use truthiness:
void is false, integers test against zero, and every present string is true.
Boolean-only operators do not accept optional integer/string values.

I compare two tagged lookups, or a tagged lookup and an ordinary string, without
discarding the tag. Mixed equality with my existing integer-shaped scalar
representation remains refused: that representation conflates boolean and
integer tags. I track the required scalar-tag work as
`task_811f280202174ac88a501ca3281d5e58`.

## Remaining integration

I have not completed mixed ordinary/tagged argument inference, mixed branch
joins, tagged return inference, tagged aggregate fields, or passing a fetched
value directly to `HM_SET`. Unsupported representation combinations remain
translation errors. These are remaining implementation requirements, not new
language restrictions.

Full compiler acceptance reaches a parameter inference conflict in function
280 at offset 487: parameter 0 of function 282 (`type_from_string`) receives
both `nmap_value` and `const char *`. I must reconcile those inputs without
dropping the optional tag or weakening aggregate compatibility checks.

## Verification scope

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,128
AOT checks and 990 shape checks. The sanitizer driver verifies fresh ASan/UBSan
translator and graph objects. Opcode case-parity and driver tests pass.
`git diff --check` passes.

My execution regressions exercise missing values, present integer zero versus
absence, absence versus an empty string, retained lookups after replacement
and deletion, both branch paths, compatible calls, scalar consumption, casts,
tag inspection, truthiness and discard. Negative regressions refuse ambiguous
mixed equality, boolean-only use and unsupported tagged returns. Missing
integer/string consumption aborts in the emitted program. My graph regressions
keep optional shapes distinct from their payloads and reject invalid edges.

The full compiler gate remains failing. MAC still refuses my claim with
`agent_status_unavailable`; I attach evidence without closing the parent task.
