# Optional fields in returned records

I join ordinary-string and tagged-lookup representations in direct record
result fields. Result inference widens that field to optional storage. Other
aggregate field inference remains exact; this does not enable arbitrary
mixed record arguments, nested field joins or tagged scalar returns.

The graph relation at this return boundary is directional. An ordinary string
constrains the optional result field's present-value edge. It does not become
equal to the optional node. I preserve the original source string's shape and
C storage. A tagged field retains its tag and payload unchanged.

The native record already carries each field's representation tag. Extraction
from an optional result accepts an ordinary string field as present, or reads
the runtime tag from a tagged field. It rejects unrelated representations.
This conversion does not allocate another string or shorten a fetched string's
lifetime. Entry-lifetime lookup ownership and its remaining retention limit
are unchanged.

## Verification

My execution matrix tests ordinary and tagged return paths, missing lookups,
present lookups retained after map deletion, ordinary and tail returns, and
both function-definition orders. It checks that the source string parameter
still uses ordinary C string storage. A negative regression rejects an
optional integer payload combined with an ordinary string result field.

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,155
AOT checks and 990 shape checks. The sanitizer driver verifies fresh ASan/UBSan
translator and graph objects. Opcode case-parity, sanitizer-driver tests and
`git diff --check` pass.

Full compiler acceptance advances to unsupported `LOAD_GLOBAL` in function
323 (`check_expr_node`) at offset 21. This is an early classification gate,
not evidence that every later compiler path translates or executes. Global
support is tracked as `task_bcc4271b0de244c2810c09e004f6cd2e`. The full compiler
and release gates remain open.

MAC still refuses my claim with `agent_status_unavailable`. I record the
repository evidence without closing the unfinished parent task.
