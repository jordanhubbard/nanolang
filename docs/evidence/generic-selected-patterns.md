# I retain concrete types through selected patterns

I validate the complete field set and selected variant identity for ordinary
generic patterns such as `let Box.Some { value } = payload`. My C checker
copies the selected binding's complete `TypeInfo` onto the hidden capture and
retains resolved metadata on inferred projections. Each AST owns its copy;
environment symbols borrow it. My existing payload resolver substitutes the
concrete fields before checking their uses.

My self-hosted checker preserves the selected binding's concrete arguments on
the hidden alias. Inferred union projections retain their full concrete type
spelling in the AST used by native emission. I capture a validated destructured
payload once with `__auto_type`, preserving its selected native layout.

My executable controls cover `Box<int>`, `Box<string>`, empty variants and
`Box<Result<int,string>>` with both integer and string selected payloads. Missing,
duplicate and wrong-variant fields, a substituted type mismatch and generic
owned transfer remain rejecting controls. Negative cases preserve the previous
output artifact. Generic owned transfer, resource collections, first-class
function signature metadata and global resource ownership are not implemented
by this change. Direct nested field scrutinee inference is separate from these
explicit selected-pattern projections.

Task: `task_bbda7f126bda403aa74034a762930f24`.

Source checkpoint `81cc5ebd` includes main through PR457. It passed a fresh
three-stage bootstrap, all C typechecker tests and 59 new/adjacent methods in 176.560 seconds
(58 paired across all three compilers and one C-only constructor-context
method; the corresponding self-hosted context repair remains separate). The combined methods include eight new pattern
methods, instantiated ownership, nongeneric selected patterns and transfers,
generic identity, and the native nested-generic constructor controls.

An isolated ASan/UBSan parser/typechecker harness passed 160 checks: eight
ordinary/negative fixtures repeated twenty times with AST and environment
teardown. Leak detection was disabled under existing metadata-leak task
`task_00c47a5d65d04c48914864ec0de553d6`; I do not claim leak freedom.

Logs: `/tmp/nanolang-generic-pattern-integrated-bootstrap.log`,
`/tmp/nanolang-generic-pattern-integrated-paired.log`,
`/tmp/nanolang-generic-pattern-integrated-typechecker.log`, and
`/tmp/nanolang-generic-pattern-integrated-asan.log`.

Selected generic ownership transfer follows under
`task_08428ceb1d674de49383aab1ba9a78c8`; this evidence does not close it.
