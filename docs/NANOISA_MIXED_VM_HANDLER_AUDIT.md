# My private mixed VM handler ownership audit

I record the actual shared-handler audit before executing the new private route.
Task `task_f1307e3b122a4d7a96a5a52a9479f672` preserves the e181 static finding:
I64/F64/BOOL handlers can pop heap owners before a tag refusal without releasing
them. The corrected private preflight checks those exact scalar tags before any
pop. I do not execute the unfixed cases or change ordinary/public behavior.

Every preflight first validates stack availability and physical heap shape.
Failures leave operands rooted for one private unwind. This table includes all93
closed opcode recipes; the other163 numeric slots remain unsupported. Implicit
completion uses the same rooted return count/tag checks as RET. Aliased globals
can mutate a prior result graph under committed effects; preserving the prior
result means preserving its owner, not rolling back arbitrary graph mutations.

| Opcode | Actual root/error handling |
| --- | --- |
| `PUSH_I64` | No consumed heap roots; fixed capacity checked before publication. |
| `PUSH_U8` | No consumed heap roots; fixed capacity checked before publication. |
| `PUSH_F64` | No consumed heap roots; fixed capacity checked before publication. |
| `PUSH_BOOL` | No consumed heap roots; fixed capacity checked before publication. |
| `PUSH_VOID` | No consumed heap roots; fixed capacity checked before publication. |
| `PUSH_STR` | Retain overflow checked before publication; source root remains live. |
| `NOP` | No consumed heap roots; fixed capacity checked before publication. |
| `I64_ADD` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_SUB` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_MUL` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_DIV_S` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_REM_S` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_NEG` | Private exact INT preflight before pop; scalar-only success. |
| `F64_TO_BITS` | Private exact FLOAT preflight before pop; scalar-only success. |
| `F64_FROM_BITS` | Private exact INT preflight before pop; scalar-only success. |
| `F64_ADD` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_SUB` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_MUL` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_DIV` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_NEG` | Private exact FLOAT preflight before pop; scalar-only success. |
| `CAST_INT` | Preserve existing conversions; heap branches release input, and failing float-to-int branch contains only a scalar. |
| `CAST_FLOAT` | Preserve existing conversions; heap branches release input, and failing float-to-int branch contains only a scalar. |
| `CAST_BOOL` | Preserve existing conversions; heap branches release input, and failing float-to-int branch contains only a scalar. |
| `CAST_U8` | Private INT-or-U8 preflight; existing handler also releases wrong-tag consumed owners. |
| `TYPE_CHECK` | Preserve existing conversions; heap branches release input, and failing float-to-int branch contains only a scalar. |
| `EQ` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `NE` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `LT` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `LE` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `GT` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `GE` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `I64_EQ` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_NE` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_LT_S` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_LE_S` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_GT_S` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `I64_GE_S` | Private exact INT/INT preflight before either pop; scalar-only success. Ordinary enum compatibility is unchanged outside this enum-free private domain. |
| `F64_EQ` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_NE` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_LT` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_LE` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_GT` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `F64_GE` | Private exact FLOAT/FLOAT preflight before either pop; scalar-only success. |
| `BOOL_AND` | Private exact BOOL/BOOL preflight before either pop; scalar-only success. |
| `BOOL_OR` | Private exact BOOL/BOOL preflight before either pop; scalar-only success. |
| `BOOL_NOT` | Private exact BOOL preflight before pop; scalar-only success. |
| `AND` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `OR` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `NOT` | Preserve existing polymorphic comparison/truthiness; existing handler releases every consumed owner. |
| `STR_LEN` | Existing handler releases all consumed roots on success and tag refusal; preserve index fallback semantics. |
| `STR_CHAR_AT` | Existing handler releases all consumed roots on success and tag refusal; preserve index fallback semantics. |
| `ARR_LEN` | Existing handler releases receiver on successful or failed tag check. |
| `STR_EQ` | Existing handler releases all consumed roots on success and tag refusal; preserve index fallback semantics. |
| `STR_CONTAINS` | Existing handler releases all consumed roots on success and tag refusal; preserve index fallback semantics. |
| `STR_STARTS_WITH` | Existing handler releases all consumed roots on success and tag refusal; preserve index fallback semantics. |
| `STR_ENDS_WITH` | Existing handler releases all consumed roots on success and tag refusal; preserve index fallback semantics. |
| `CAST_STRING` | Existing string input transfers its owner; scalar branches own no heap input; default releases input before allocation. |
| `STR_CONCAT` | Existing handler releases consumed roots and scratch on all allocation/tag outcomes; private interning checks refcount overflow before retaining aliases. |
| `STR_SUBSTR` | Existing handler releases consumed roots and scratch on all allocation/tag outcomes; private interning checks refcount overflow before retaining aliases. |
| `STR_TRIM` | Existing handler releases consumed roots and scratch on all allocation/tag outcomes; private interning checks refcount overflow before retaining aliases. |
| `STR_TO_LOWER` | Existing handler releases consumed roots and scratch on all allocation/tag outcomes; private interning checks refcount overflow before retaining aliases. |
| `STR_TO_UPPER` | Existing handler releases consumed roots and scratch on all allocation/tag outcomes; private interning checks refcount overflow before retaining aliases. |
| `STR_REPLACE` | Existing handler releases consumed roots and scratch on all allocation/tag outcomes; private interning checks refcount overflow before retaining aliases. |
| `STR_FROM_INT` | Existing handler releases consumed roots and scratch on all allocation/tag outcomes; private interning checks refcount overflow before retaining aliases. |
| `STR_FROM_FLOAT` | Existing handler releases consumed roots and scratch on all allocation/tag outcomes; private interning checks refcount overflow before retaining aliases. |
| `STR_SPLIT` | Existing partial-array cleanup releases every retained segment and input; private string creation/push guard each retain before publication. |
| `LOAD_LOCAL` | Retain overflow checked before publication; source root remains live. |
| `LOAD_GLOBAL` | Retain overflow checked before publication; source root remains live. |
| `STORE_LOCAL` | Destination checked before pop; move new owner, then release previous owner. |
| `STORE_GLOBAL` | Destination checked before pop; move new owner, then release previous owner. |
| `DUP` | Retain overflow checked before publication; source root remains live. |
| `SWAP` | Reorder existing roots without retain or release. |
| `POP` | Pop and release exactly once. |
| `ASSERT` | Move condition to trap; private supervisor releases it on either truth outcome. |
| `JMP` | No consumed root; exact branch edge checked during preparation. |
| `JMP_TRUE` | Existing truthiness semantics; release consumed condition on either branch. |
| `JMP_FALSE` | Existing truthiness semantics; release consumed condition on either branch. |
| `CALL` | Check argument tags/frame capacity before transfer; same stack roots become callee locals. |
| `RET` | Check result count/tag while rooted; stage result, release locals, publish after frame removal. |
| `STRUCT_NEW` | Private exact descriptor/count/child validation before allocation/pop; map compact ordinal to global layout, then move owners. |
| `STRUCT_LITERAL` | Private exact descriptor/count/child validation before allocation/pop; map compact ordinal to global layout, then move owners. |
| `AGG_PACK` | Private exact descriptor/count/child validation before allocation/pop; map compact ordinal to global layout, then move owners. |
| `ARR_NEW` | Private exact flat element tags before allocation/pop; successful literal moves owners into completely allocated storage. |
| `ARR_LITERAL` | Private exact flat element tags before allocation/pop; successful literal moves owners into completely allocated storage. |
| `STRUCT_GET` | Receiver/bounds and retained child checked while rooted; retain child, release receiver; missing array read preserves VOID semantics. |
| `AGG_GET` | Receiver/bounds and retained child checked while rooted; retain child, release receiver; missing array read preserves VOID semantics. |
| `ARR_GET` | Receiver/bounds and retained child checked while rooted; retain child, release receiver; missing array read preserves VOID semantics. |
| `STRUCT_SET` | Exact nominal/element destination preflight; move new edge before releasing old edge; receiver owner survives. |
| `AGG_SET` | Exact nominal/element destination preflight; move new edge before releasing old edge; receiver owner survives. |
| `ARR_SET` | Exact nominal/element destination preflight; move new edge before releasing old edge; receiver owner survives. |
| `ARR_PUSH` | Exact element and retain overflow before handler; allocation completes before append/retain; handler releases staged value. |
| `ARR_POP` | Transfer removed child owner; release receiver after removing edge. |
| `ARR_SLICE` | Existing handler releases receiver and both endpoint owners on every outcome; private boxed-copy overflow rolls back only completed retains. |
