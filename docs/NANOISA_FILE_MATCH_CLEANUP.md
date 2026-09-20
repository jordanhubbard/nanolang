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
