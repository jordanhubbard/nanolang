# My typed array receiver mutation

I infer the complete native array receiver type before selecting `array_set`.
Record field projections, nested projections, and array-returning calls now
select the matching setter instead of defaulting to integer storage. I pass
the element type when lowering the replacement expression, including empty
nested arrays. My existing generic fallback remains unchanged for unknown
receiver forms.

My fresh three-stage bootstrap and six existing array compatibility methods
pass. The new regression runs bool/string/float fields, nested record fields,
returned arrays and nested-array replacements. Eleven of twelve compiler/case
combinations pass. The remaining C-seed nested-array literal case fails with
an incompatible pointer before executing the setter; I track that prerequisite
as `task_c2ebfd28c24345daaa8c31dac75b45ac` and retain the failing assertion.
This checkpoint is not ready to merge until that case passes as well.

I track the receiver repair as `task_d32adbdff13241dc8ad9b0a889071352`.
