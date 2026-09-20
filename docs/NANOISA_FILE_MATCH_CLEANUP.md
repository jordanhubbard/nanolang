# I repair the measured zero-payload match cleanup boundary

I retain the5a534 Linux GCC and puck Homebrew sanitizer terminals. Each reports
162 bytes in12 allocations after my ordinary/parser assertions complete. My
ordinary two-method and corresponding publisher suites passed before those
terminals. This design belongs to task_63edb399b8cd44ef82eaffa99328a22e under
8bbc; it does not establish general union ownership or File source execution.

## I distinguish allocation from an alias

My interpreter currently represents unions with uncounted pointers. An
identifier, parameter, field, call, conditional or nested match can return an
alias. I cannot free every scrutinee or add recursive union destruction to
`env_free_value` without first defining that larger ownership model.

For this correction I recognize only an immediately constructed, empty union:

- The scrutinee AST is `AST_UNION_CONSTRUCT` with zero supplied fields, or
  `AST_STRUCT_LITERAL` with zero supplied fields whose evaluated value is a
  union. The latter is my actual `Choice.None {}` parser representation.
- Its evaluated `UnionValue` has zero fields and null field-name/value arrays.
  Both actual constructor branches allocate a new union header and duplicate
  its two identity strings. They do not evaluate any payload expression.
- No pattern can obtain this union pointer: a named empty payload receives
  `create_void`, while empty/discard/wildcard bindings create no payload local.
  Existing caller aliases and nested/payload-containing unions are outside
  this predicate and retain their existing ownership behavior.

I retain this precise owned-temporary flag across guard fallthrough. On the
successful arm or propagated guard control exit I release only the owned
header and its two identity strings. I preserve an exact union-pointer-equal
escaping result defensively rather than release it. A returned independently
constructed union is a distinct pointer and remains the caller's result.
I do not release an unknown scrutinee, child payload or borrowed root.
Process-exit invariant failures remain process exits; I do not claim general
recoverable interpreter failure cleanup.

## I retire match metadata without retiring borrowed type facts

The interpreted arm's binding symbol owns its duplicated `name` and optional
`struct_type_name`; its `type_info` points into retained declaration facts.
I free those two owned strings before each existing saved-symbol-count reset,
including a false guard and a propagated guard result. Body-local cleanup
already happens inside the arm block. I do not free a binding's runtime value
or shared type facts in this metadata-only operation. Empty named payloads are
void; general nonempty payload lifetime remains a separate requirement.

Native generation appends temporary parameter/local/restored-match symbols.
Their names and optional nominal strings are owned copies; AST names,
`def_file`, `type_info` and declaration roots are borrowed. I add a local
metadata-pop helper in `transpiler.c` and use it for all three function-level
restorations: ordinary completion, open-record stub and generic completion.
I call it only after `transpile_statement` has processed/freed its work list
and lexical scope stack and copied emitted text into the output builder.
I audit any retained registry/capture pointers before implementation; any
reference that outlives this point must retain its own copy or block the edit.
I do not change the environment layout, value destructor or symbol ABI.
The symbol index stores indices/hashes, and already supports freeing names
before lowering the count; I preserve that contract.

## I qualify the boundary without hiding the larger requirement

I preserve every existing parser/ordinary execution assertion and strict leak
checks. I add controls for repeated zero-literal matches, named-empty and
empty/discard/wildcard arms, false-guard fallthrough, early return and distinct
returned union ownership. Caller-owned alias and nested/payload-union controls
must show that the new cleanup does not release them; their fixture owner
explicitly frees only storage it constructed. This manual owner does not
establish a general interpreter union lifetime policy.

I retain full union/borrow/escaping-payload lifetime as an open5.1 requirement
under the parent ownership/source work. The bounded correction may establish
only the fresh-empty-scrutinee and temporary-symbol metadata boundary.
Fresh C seed and Stage1/Stage2 qualification is required because the actual
interpreter/native compiler changes. Sanitizer inventory will additionally
name freshly instrumented changed interpreter/transpiler providers, retaining
ordinary common-provider limits. Source and fixtures require review before
execution. Existing successful5a534 ordinary/source/generator/publisher facts
remain at their original pin; neither failed sanitizer is relabeled.

## I audit the production checkpoint's allocation and retained pointers

| Boundary | Allocation or retained reference | Retirement |
| --- | --- | --- |
| `eval_expression`, dotted `AST_STRUCT_LITERAL` union branch | With zero fields I call `create_union` directly after exact variant arity checking; no child expression is evaluated | New match predicate checks literal kind, zero supplied/actual fields and both null arrays |
| `eval_expression`, `AST_UNION_CONSTRUCT` | With zero fields both local arrays remain null and I call `create_union` once | Same predicate; every nonliteral scrutinee is excluded |
| `create_union` in `env.c` | `malloc(sizeof(UnionValue))`, `strdup(union_name)`, `strdup(variant_name)`; zero-field arrays are explicitly null | Three frees only, unless the exact pointer escapes as a union result |
| `env_define_var_with_type_info` | Duplicated symbol name and possibly duplicated inherited nominal name | Interpreted match metadata pop frees those names; value/type facts are untouched |
| `restore_native_match_binding` | New nominal name plus the environment's duplicated binding name | Native function metadata pop after complete statement emission |
| Native ordinary parameters and emitted locals | Parameter nominal `strdup`, local owned nominal `strdup`, environment-owned names | Same pop; saved preexisting symbols remain live |
| Native generic parameters | Environment-owned name/optional inherited nominal, copied scalar type fields | Pop after generic body emission; no value destructor |
| Open-record stub parameters | Same ordinary parameter ownership, but no body work list | Pop after stub text append |

My two constructor routes allocate fresh headers through `create_union`; they
never reuse an input union pointer. Its legacy allocation-failure handling is
unchanged and is not qualified as recoverable by this correction.

`emit_literal`, formatted work items and GC-release work items retain their own
strings. `scope_add_var` duplicates names. `transpile_statement_iterative`
processes the work list, frees it, and frees its scope stack before returning.
The output StringBuilder has copied text at all three pop sites. Function and
tuple registries retain signatures/type facts, not the popped symbol's name or
nominal string; those signatures and type facts remain owned by declarations.
Effect capture arrays are used during nested emission and restored to their
outer context before function emission returns; the function-level pop does
not run during a live nested work list. This audit does not repair the separate
effect-parameter metadata reset inside the iterative emitter.

I leave runtime Values, `type_info`, `def_file`, source positions and preexisting
symbols untouched. I null the two retired metadata pointers before lowering
the count. My environment index uses slot indices and hashes, so it needs no
freed name when synchronizing a shorter count. No allocator is added to the
new cleanup path, and no native storage or ABI field changes.

## I prepare the cleanup fixture without executing it

My existing208 persistent/transient allocation rows, ordinary match controls,
complete actual publisher parse/refusals and paired947-shadow corpus remain.
I add two separately named output rows, and the Python harness requires every
row exactly once in its fixed order.

`tests/file_service_parser_eval.c` includes my actual interpreter in a fresh
selected provider. Its direct helper controls distinguish both literal AST
kinds from identifiers/calls/fields/conditionals/matches, check supplied payload
and actual payload exclusion, preserve an exact escaping pointer, and preserve
a distinct returned union. Its alias controls invoke the actual match evaluator
16 times each over caller-owned empty, scalar-payload and nested-union roots.
The fixture owns/frees those roots explicitly after checking pointer/content
identity; this does not qualify general interpreter union destruction.

Four real parsed/typechecked ordinary source programs exercise false-guard
fallthrough, wildcard/discard, named-empty binding and a distinct returned
empty union. `choose` returns the tested value; `main` remains an ordinary int
entry. Each program executes32 times, emits native C four times, and compares
all preexisting symbol names/nominal names and symbol counts afterward. I free
the explicitly returned empty union as the host fixture's result. A separately
labeled direct-AST guard-return control invokes the real evaluator over the
parsed fresh literal while preserving its original guard table afterward.
The exact defensive same-pointer helper path is a unit control, not a claim
that ordinary zero-field syntax can expose that scrutinee pointer.

I freshly compile actual `eval.c` through the fixture wrapper and `transpiler.c`
with the same ordinary or ASan/UBSan flags as the parser/env/lexer/UTF8 providers.
Their old ordinary objects are rejected from the shared object list. The actual
iterative emission provider remains ordinary and explicitly inventoried; this
is not full-compiler instrumentation. Generic/open-record metadata-pop sites
retain the static lifetime audit; the new repeated native source controls
exercise ordinary function completion directly.

I require a fresh full compiler bootstrap on each host before the complete
paired/schema corpus, seven C configurations, actual publisher/strict binding
adjacency and parser/module/wrapper neighbors. I retain all successful5a534
ordinary phases and both first LSan terminals separately. No new gate runs
before review of this full fixture checkpoint.
