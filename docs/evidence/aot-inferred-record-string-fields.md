# My inferred record string fields

My flat field facts describe known incoming values, not every nested caller.
A projected record can supply an optional string after flat classification has
seen only a direct caller's string. I previously turned that incomplete fact
into an exact constraint on the shared parameter and its field projection.
My final solver then refused the optional-to-exact-string conversion.

I now seed inferred string field facts with directed storage flows at record
boundaries and field projections. My explicit constructors and native array
and map payload constraints remain exact. My solver still rejects a known
integer payload where a string payload is required. I do not change its
exact-destination rule.

Four regression cases cover both function orders and ordinary/tail calls.
Each combines a direct string-record caller with a projected optional-record
caller and executes present and absent values. All four failed translation
before this change and now execute in NanoVM and native code. Four companion
negative cases reject a known integer map payload even when the lookup is
absent.

My invalid string-as-integer return fixture still fails before execution.
Its error now comes from deferred conversion solving rather than the `RET`
classification step; I accept only that specific alternative diagnostic.

My normal and fresh ASan/UBSan suites each pass 1,670 AOT and 1,073 shape
checks. Leak detection is disabled; these runs do not establish leak freedom.

Fresh compiler acceptance passes six focused test methods. The full compiler
now passes storage-conversion solving, including the `env_get_type` (311)
conflict at offset 268, but fails later: `type_from_kind` (260), offset 165,
has an unresolved `AGG_PACK` field 0. I track that failure as
MAC `task_041fdf3407774b93a24bbaaa30a7bbb9`. Full compiler and release acceptance
remain incomplete.

MAC still refuses my claim for `task_146d0626844a4b958bfe0e8697226185` with
`agent_status_unavailable`. I retain evidence without forcing closure.
