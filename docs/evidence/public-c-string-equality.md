# My public C string equality evidence

I tested reviewed production7c22c401 with frozen harness3696124c after canonical
PR749 integration. My [manifest](public-c-string-equality.json) records thirteen
source/harness/tool hashes verified unchanged after all gates and eight retained
logs. My [contract](../PUBLIC_C_STRING_EQUALITY_CONTRACT.md) precedes implementation
under task4c6d924e0aa242df950f662f61981664.

I compare exact STRING operands by content, preserving identical-pointer/null
handling. Distinct automatic pointer slots and comma sequencing evaluate left
before right exactly once. Nested comparisons, recursive frames, function bodies
and ordered global initialization retain separate storage; branches and loops
keep evaluation at the original expression. A STRING paired with UNKNOWN or a
non-STRING receives a checked refusal before publication. Other scalar comparison
rules stay unchanged.

| Frozen gate | Result |
| --- | --- |
| New equality GCC / Clang | 4 / 4 methods, 1.449 / 1.901 seconds |
| Formatting GCC / Clang | 4 / 4 methods, 1.238 / 1.733 seconds |
| Scalar/API GCC / Clang | 8 / 8 methods, 3.943 / 4.592 seconds |
| Existing public C backend | 7 programs pass, no skips |

My new ordinary source covers distinct conversion allocations, aliases, scalar
record fields, empty/unequal text, nested effectful operands, globals, both branch
selection and loop reevaluation, and namespace collisions. Actual public C executes
C99/C11 O0/O2 with ASan/UBSan/default Linux leak checks. One independent source also
passes the interpreter, verified VM and sanitized native C route. Generated-helper
controls check pointer/null cases directly; eight mixed/unknown operator-side API
combinations retain path/stream output before a same-process valid recovery.
Adjacent scalar controls retain initializer reentry and arithmetic behavior.

I did not replay the historical formatting assertion binaries. Those source/logs
remain with PR749's evidence. This corrected source uses fresh independent programs.
No bootstrap or Darwin qualification is claimed for this C-only change.

I qualify existing NUL-terminated public C strings, not arbitrary length-bearing VM
byte strings. Static audit separately records raw source escape transport under
cf8714a41ada4272b26da1db3b6450c2: canonical source decodes escapes before strlen,
whereas public C currently re-escapes raw parser text. That required correction,
concat ownership, remaining GNU conversions/blocks and full6ade remain open.
