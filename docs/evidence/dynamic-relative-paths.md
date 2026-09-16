# Dynamic relative-path storage

I remove the bounded private normalizer, token arrays and output buffer from
`path_relpath`. I normalize both inputs into owned dynamic buffers, compare
components in order, and allocate a checked upper bound for the result. I free
both intermediate buffers on success or allocation failure. Allocation/size
failure returns NULL rather than a truncated path.

## Verification

`make test-path-normalize` passes with the complete filesystem module linked.
New relative-path cases cover ordinary shared prefixes, parent results,
identical long paths, a shared prefix beyond 256 components, a 5,000-byte target
and 1,500 base components producing a 4,504-byte result. Existing generated-native
and module normalization cases remain passing.

`make test-directory-walk` passes six cases in 3.933 seconds, with one host
path-limit skip. `make test-nvm2c` passes 828 checks. `git diff --check` passes.

This completes the storage-bound work in MAC
`task_82bd388637824cc889b12204d226e75b`. MAC still refuses my worker claim as
unavailable, so repository completion does not imply ledger closure.

I preserve the existing lexical component comparison rules here. Inspection
also identifies a separate semantic defect: dot is treated as a component,
and mixed roots or unresolved parents lack a shared anchor contract. I track
that work under `task_64696231d8984732a5a1e1c319ca043b`; removing size limits
does not establish correctness for those inputs. Other filesystem helpers
still have independent limits. Full compiler acceptance and release remain open.
