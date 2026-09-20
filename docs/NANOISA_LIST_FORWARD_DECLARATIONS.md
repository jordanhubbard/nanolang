# My deferred list-field declarations

I track this prerequisite as task_addf07bd93224ebca3af7c7479a578c4 under my
required list/evaluator acceptance. Both b915 make-build attempts failed before
bootstrap and fixtures; my first Linux and Darwin reports remain respectively
/tmp/nanolang-record-lists-b915-linux-prepare and
/private/tmp/nanolang-record-lists-b915-puck-prepare. Their source/tool maps were
unchanged. The failure is eager checker registration, not measured ABI execution.

I first collect each module's declarations. Then I revisit stored list fields
under their declaring module. A resolved field registers its exact ordinal. An
unresolved field in an extern declaration remains declaration-only and is
revisited as importing modules finish collection. I create no placeholder list
specialization or callable for it. Ordinary unresolved fields and every actual
list value boundary still refuse; I distinguish missing identity from a genuine
generated-name collision.

My existing foreign contract is explicit: bind_nominal_records in
src/nominal_types.c leaves extern record names unmangled and rejects collisions
between foreign declarations or a foreign and ordinary declaration. My generated
compiler_contracts.nano intentionally refers to LexerToken before its importer
compiler_ast.nano declares that extern C type. I may resolve that unqualified
foreign name to its sole already registered is_extern StructDef ordinal, retaining
its original module. I do not infer ordinary record identity from unique spelling.
A competing declaration with the same actual or original name refuses. Qualified
aliases still require their existing exported-name and exact module checks.

I retain both checker registration routes and existing value-use checks. I test
forward fields within a module and across an importer, both import orders, exact
extern identity and alias resolution, colliding ordinary/extern declarations and
unresolved actual use. Declaration-only foreign fixtures do not claim foreign C
ABI execution. My unchanged compiler graph remains the real bootstrap gate.

I require source/fixture review before corrected execution. The failed b915
source remains frozen and no later pass replaces either first terminal.
