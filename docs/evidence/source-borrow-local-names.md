# Borrowed-source lexical names

I extend my optional `nano.local.v1` convention under
MAC `task_0178be8daf554a69969144b6657e7756`, after source admission PR #599.
My contract commit is `2ad799b2`; producer checkpoint `aa657162` starts from
canonical main `36f87958`. The final pool-order companion is `43e2f6c4`.
My runtime and ownership verifier are unchanged.

I retain borrowed parameters at PC zero, named owners after OWN_STORE_LOCAL,
and scalar lets/projections after STORE_LOCAL. I omit constructor scalar
temporaries and the parser-generated destructuring owner. A moved owner's
name remains a lexical fact, never a promise of liveness. Selected shadows
end their names after cleanup; following shadows receive distinct slots.

My tests compare C, Stage1 and Stage2 names, validate exact instruction-boundary
intervals, roundtrip canonical text twice, remove optional metadata, and execute
both annotated and stripped modules in VM and sanitized native code. A small C
harness exposes codegen_compile_shadows for direct comparison against the three
selfhost shadow emitters. Existing source-borrow refusal/publication tests remain
in the same gate. The broad reconstruction and ownership parents remain open.

My fresh default three-stage bootstrap passed at this source checkpoint:
`/tmp/nanolang-borrow-local-names-bootstrap.log`. The corrected target at
`43e2f6c4` passed a fresh default three-stage bootstrap and all seven source-borrow
methods in 138.731 seconds: `/tmp/nanolang-borrow-local-names-final.log`.
This includes exact full dumps from both producers, both canonical native
compiler stages, passing/failing shadow publication, C/selfhost selected-shadow
names, and annotated/stripped VM plus ASan/UBSan/LSan native execution.
The same target passed 123 codec checks, five local-name allocation boundaries
and twenty marker allocation boundaries. The unchanged ordinary-name producer
gate passed both methods in 0.420 seconds:
`/tmp/nanolang-borrow-local-names-adjacent.log`.

My first seven-method draft gate retained fourteen failed comparisons in
`/tmp/nanolang-borrow-local-names-paired.log`: twelve full-dump comparisons
exposed the helper function name being interned after entry name records by my
selfhost assembler, and two new strip comparisons included empty separator
lines. I predeclare both function names before records and compare every
nonempty canonical line when stripping; the complete producer dump equality
checks remain exact. My corrected shared fixture's complete dumps compare
byte-for-byte. I retain this draft evidence rather than calling it infrastructure.
