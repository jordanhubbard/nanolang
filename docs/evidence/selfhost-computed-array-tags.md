# My computed array element tags

On 2026-09-16 I reproduced a direct literal `[(int_to_string 7)]` emitting
`CAST_STRING` followed by `ARR_LITERAL 1 1`. Executing the old module failed
its assertion that the first element equaled `"7"`. My literal lowering had
checked AST node kind instead of expression type.

I now infer homogeneous supported scalar element types and use the same
declared-context literal lowering already used for typed bindings. Computed
strings retain tag 5; integer expressions retain tag 1. Mixed, nested and
unsupported element kinds refuse output. Empty context-free literals retain
the existing integer default.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 26 integration methods.
The added fixture contributes six C-seed bytecode comparisons and executes
under NanoVM and strict C11 AOT, covering computed strings, both orders of
literal/computed string elements and computed integers. Four malformed literal
forms refuse output. The measured full-compiler boundary remains nested
`List<CompilerDiagnostic>`; this repair does not claim bootstrap completion.
