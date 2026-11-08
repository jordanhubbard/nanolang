# My existing scalar array sort contract

I preserve `tests/test_stdlib_arrays.nano` unchanged. My existing C/runtime builtin clones its input storage before sorting and returns the same scalar element kind. `dyn_array_sorted` supports INT, U8, FLOAT, BOOL and STRING; other families return no sorted array. Source/native admission must establish the flat element storage and retain it in the result rather than accepting an arbitrary ARRAY tag.

I compare integers and bytes without subtraction overflow, put false before true, and compare string bytes with `strcmp`. My float comparator puts numbers before NaNs and compares ordinary numbers by numeric order. Positive and negative zero compare equal; NaNs compare equal to other NaNs. I preserve their stored payload bits but do not promise stable order among equal comparisons. A different signed-zero or NaN payload order is not by itself a sorting defect.

I evaluate the receiver once. Sorting does not mutate the input array, and subsequent element replacement or growth of either array must not change the other's storage. My typed empty inputs retain their element identity. I preserve existing lexical function bindings and wrong-arity/type refusals before replacing prior output. My STRING-array foreign bridge's byte/UTF-8 boundary remains part of the contract; ordinary UTF-8 examples do not establish embedded-NUL equivalence.

I first retain the current source/VM/native baseline and inspect the actual VM/runtime ABI. Exact builtin lowering, actual Nano checker controls, sanitized generated programs, lifetime/root controls and fresh-stage/platform acceptance follow under #1109/#1081. I do not substitute a new generic host or artifact descriptor for this existing builtin.
