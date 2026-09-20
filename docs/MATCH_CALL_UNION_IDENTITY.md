# Checked generic union call identity

I track this correction in `task_a1d38d93616f4c48a3b8692b30226a82`.
My baseline is `af8809b32850454d06d9c1881c3a27b43f4c9d9c`.

I retained arm64 CI runs 35498179116 (PR 906), 35498613035 (PR 907),
and 35498722005 (PR 908). Each completes bootstrap stages 1/2/3 and then
rejects `scripts/userguide_snippets_check.nano:201` with E001 when a match
scrutinee calls `extract_json_from_marker`, declared `Result<string, string>`.
The raw job logs, API records, exact failure excerpts and hashes remain in
`/tmp/nanolang-arm64-ci-906-908-audit`. I have not replayed failed binaries.
These are observed compiler refusals, not attributed infrastructure failures.

Both C checker match paths read a call's `return_struct_type_name` but omit
its declared `return_type_info`. My existing checked-expression metadata
reader already resolves direct function, checked callable and field return
metadata. I will use that reader for call scrutinees just as I do for fields,
retain the registered generic base for coverage, and derive the concrete
instance name and payload metadata from its exact type arguments. I borrow
this metadata; I do not introduce an owner or change its lifetime.

I retain unresolved-identity refusal, exact BOOL guards, domain checks,
coverage, purity and resource rules. I do not substitute a variant spelling
for a missing scrutinee identity or rewrite the original failing source.

Before execution I require independent source and fixture review. My focused
controls cover direct generic Result and Option calls in statement and value
matches, two different payload instantiations, actual call-once traces,
conditional guards, wrong payload use and missing-arm output preservation.
I retain the complete existing C-seed match-totality controls, including the
unresolved nested match refusal. Fresh acceptance also compiles the unchanged
user-guide snippet checker and runs its existing check target, then the
relevant shared match controls and fresh bootstrap. Platform and provider
attribution stay explicit; this correction does not close full generic-union,
affine, bootstrap-equivalence or release acceptance.
