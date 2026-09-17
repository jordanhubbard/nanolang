# My enum declaration metadata

I retain ordered `variant_names` and `variant_values` arrays alongside
`ASTEnum.variant_count`. My parser records signed explicit values and continues
implicit numbering from the preceding value. Empty and separate declarations
retain separate arrays. The generated NanoLang and C layouts share these fields.

A fresh three-stage bootstrap and schema gate pass. My parser harness compiles
and executes with C seed, Stage 1 and Stage 2, checking zero defaults,
noncontiguous values, negative values, duplicate numeric values, implicit
continuation and empty/separate declarations. Mandatory parser shadows also
check the stored source location and metadata.

This supplies parsed facts for NanoISA enum lowering. It does not itself add
enum bytecode lowering or complete compiler bootstrap. I track this prerequisite
as `task_e66b50097fe343e3b78e6b750a5c7315`.
