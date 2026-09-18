# My managed byte-predicate evidence

I implement required target-coverage child
`task_b9a83a6b239b4c858ca6324d3066ac33` from pre-code contract `174ccb7f`.
My frozen production checkpoint is `dec6b1ba` on main `4f715b09`.

I compare exact stored bytes for CONTAINS/STARTS_WITH/ENDS_WITH, including NUL
and high bytes. Empty needle/affix succeeds; oversized inputs fail the predicate.
A borrowed helper validates both views and the selector, allocates nothing and
publishes output only on success. The nonempty-needle branch bounds the final
search offset below UINT32_MAX. My LLVM tag guard and ordinary FrameOutput
cleanup release both consumed operands, including dynamic type-error paths.
Only the managed profile gains these operations.

My 18 paired inputs exercise empty/equal/long/absent/interior/prefix/suffix,
embedded-NUL/high-byte, overlapping and 1,025-byte search cases. All three
predicates run on literal and computed handles, through actual VM and emitted
native LLVM/import-free Node/Wasmtime modules. Same-handle aliases, helpers,
global replacement, repeated entry and terminal disposal retain one global
owner then reclaim it. Six dynamic wrong-tag cases cover both operand orders
and preserve the surviving global while cleaning consumed/frame owners.

My standalone runtime controls disable allocation and check unchanged counts
and two owners per input after predicates. They also verify unchanged output
and reference state after invalid-selector/null-output refusal. Native core and
emitted IR sanitizer checks, import-free Wasm links and Node/Wasmtime runs use
the real runtime. Initial three focused methods passed in 4.539 seconds; direct
literal cases and null-output coverage were then included in the final gate.

The initial implementation build caught two mistaken local identifier names
in the new module wrapper; I corrected them before any target execution.
The first adjacent gate also found an old translator-refusal fixture using
STR_CONTAINS, which this contract now admits. I changed old translator-refusal
fixtures to still-unsupported trim/lower operations and retained explicit
scalar/literal API refusal for all three predicates in the shared-profile gate.
I preserve the earlier logs without presenting them as successful validation.

My managed parent51da remains open. Aggregate/cycle488, host linkage2d2,
Darwin managed sanitizer7ba and historical evaluator791a remain separate
requirements; these predicates do not establish full runtime/release acceptance.

My frozen corrected gate passes 11 global methods, nine literal methods, two
package methods, three runtime-core methods, 25 managed methods and the shared
profile/refusal checks. The log is
`/tmp/nanolang-managed-predicates-integrated-r2.log`; the managed group completes
in 24.585 seconds. This includes parser/formatter/substr/concat and existing
allocation-failure/cleanup regressions alongside the new predicates.

I restacked cleanly onto main `cc9c7a1d` through PR663. Restacked production is
`3a23f0a7`. A direct Git comparison confirms all six changed production files
and the four affected test files are byte-identical to the frozen passing
`dec6b1ba` checkpoint. Incoming reconstruction scripts/docs introduce no change
to this runtime or verifier contract; I do not repeat unchanged acceptance.
