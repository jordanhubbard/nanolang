# My checker module-name ownership

I retain both 45c ordinary passes and sanitizer allocation-executable failures.
All 2,167,886 annotation allocation assertions complete; LeakSanitizer then
reports 8,532 bytes in 711 allocations. The new union payload setup repeats
237 Environments, each with three 12-byte owner names. This does not establish
full bootstrap or source/native acceptance.

I distinguish three boundaries. The fixture assigns borrowed Caller after
type_check_module has installed an allocated current_module. That drops the
last direct pointer to the allocation. Separately, production checker struct
and enum collectors duplicate module_name, while Environment destruction
intentionally treats these declaration owner names as borrowed. Existing
manual producers use static names; adding blanket destructor frees is wrong.

My writer audit covers both checker collectors, effect declarations, imported
nanovirt records, evaluator enum registration, module loader context save/
restore, and temporary checker/evaluator function owners. Only the two module
declaration collector branches free current_module; other writers borrow an
AST, function, module-cache, or caller name and restore it. Both struct and
enum collectors allocate their own names. UnionDef and EffectDef retain their
separate existing owning contracts and are not changed.

I will register only newly allocated checker current_module, StructDef and
EnumDef owner strings with the existing Environment checker allocation list.
Each fresh copy is registered once. Replacing current_module must not free its
previous value: it may be borrowed, and a registered old value remains owned
by the Environment until teardown. This preserves declaration aliases and
reentrant saved contexts. General env_define_struct/enum APIs still borrow
owner names; unrelated duplicate-record storage cleanup remains open.

I found one escaping reader that also needs correction: extract_module_metadata
shallow-copies StructDef and EnumDef before duplicating other fields. Its owner
names would otherwise borrow the Environment's new managed strings. I must
copy those names into the metadata snapshot and release them in
free_module_metadata, auditing every metadata constructor for initialization
and ownership first. No snapshot may outlive its owner with a borrowed new
name. Other already borrowed annotation fields are outside this name-only
change and must not be claimed as fully owned snapshot parity.

I preserve the registry's existing allocation-failure contract: raw ownership
entry failure is fatal. The current annotation prefix sweeps start after
module setup and do not qualify recoverable allocation failure in that legacy
registry. I add borrowed prior context, repeated module declaration, retained
owner identity and snapshot-after-Environment teardown controls. Every prior
assertion and both first sanitizer terminals remain required. Source review
precedes corrected execution.

The production declaration-name boundary belongs to
task_3a42bcf1be0c46ce88a085b4cb54bf2c; observed cleanup continuation also updates
task_ebfdc333d6a8495b8c903ab8142c88b6.
