# My compatible record arguments

I now use the same source-to-destination compatibility rule for record
arguments that I use for record results. A callee field can widen from ordinary
string storage to tagged storage when another caller supplies an optional
value. I constrain the ordinary string against the optional payload, not against
the optional container. I retain incompatible-payload rejection.

I do not rewrite the caller's record or erase its field representation. My
existing record projection emits a tagged string when the runtime field is an
ordinary string, or copies the existing tag and payload when it is already
tagged. This also applies through ordinary and tail record-returning calls.

I test both caller orders and both function-definition orders, direct and tail
calls, absent and present global values, unchanged caller records, and a
conflicting string-versus-optional-integer payload. I replace the old mixed-field
rejection fixture with a runtime assertion.

I verified:

- `make -j1 test-nvm2c`: 1,259 AOT and 994 shape checks pass.
- `make -j1 test-nvm2c-sanitizers`: the same checks pass with fresh verified
  ASan/UBSan translator and shape objects. I do not claim leak freedom.
- Opcode case parity and all three sanitizer-driver unit tests pass.
- `make -j1 test-one-ir-compiler` passes the prior function 323 offset 1761
  conflict, then fails at offset 5359: parameter 0 of `strip_spaces` is inferred
  as both a string and a record.

I traced the new failure to indexing the result of `split_tuple_type_names`.
My classifier currently treats every array-returning function as returning a
record array. I must infer scalar array result kinds and emit matching native
signatures; I track this as `task_f6b029d7b7e749caa7a064c9566bd666`.

I have not completed nested or other scalar field widening, aggregate global
storage, or full compiler acceptance. The AOT parent remains open. MAC still
refuses my worker claim with `agent_status_unavailable`; repository evidence
does not imply ledger ownership.
