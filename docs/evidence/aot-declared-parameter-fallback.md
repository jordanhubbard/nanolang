# My declared parameter fallback

My bytecode records function parameter tags, but my AOT classifier previously
used only caller-derived facts. An uncalled function could therefore pack an
unresolved scalar even when its signature declared the type. In my full
compiler, `type_from_kind` (260), offset 165, exposed this missing connection.

After caller inference converges, I seed still-unknown parameters from supported
integer, boolean, string, record and union declarations, then resume inference.
I preserve already observed representations, including tagged scalar arguments.
An array or map declaration alone does not determine its element or payload
storage, so I do not invent those facts.

Eight regression cases cover integer, boolean, string and record parameters in
both function orders. Their packed helpers have no NanoISA callers, so the
signature must supply the missing fact. All eight failed translation before
this change. They now translate, and a C harness calls each helper and checks
the packed value and storage tag. The harness renames the generated entry point
to keep that native ABI test independent of caller inference.
Removing the declaration from each case still rejects the unresolved field;
I do not guess a storage type when both caller facts and declarations are absent.

Six additional cases declare scalar parameters but pass an observed tagged
absent value. They execute in both NanoVM and native code and verify that the
declaration does not erase the observed tag.

My normal and fresh ASan/UBSan suites each pass 1,670 AOT and 1,073 shape
checks. Leak detection is disabled; these checks do not establish leak freedom.

Fresh compiler acceptance passes eight focused test methods. The full compiler
passes `type_from_kind`, then fails at the uncalled `codegen_next_temp` (362),
offset 28: `AGG_PACK` field 0 remains unresolved. Its record parameter tag does
not supply a field layout. I track that work in MAC
`task_8aaa3f722ce34624a4bb8a16283afa2b`; I have not established full compiler or
release acceptance.

MAC refuses my claim for `task_041fdf3407774b93a24bbaaa30a7bbb9` with
`agent_status_unavailable`. I retain the evidence without forcing closure.
