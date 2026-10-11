# Self-hosted effect declarations and handlers

I retain explicit effect declaration, perform and handler records in my shared
compiler schema. My self-hosted lexer/parser, checker and NanoISA emitter now
handle scalar effect operations, dynamic handlers, resumption and lexical return.
My assembler resolves symbolic operation strings for HANDLER_PUSH and PERFORM.
Both schema generators produce identical artifacts.

My first full compiler build failed two handler shadows. The diagnostic AST showed
unset block kinds for handled bodies and arms. I now mark those references as
blocks and assert their kinds, including nested handlers. All 1,136 shadows pass
in the corrected seed build. I also check resumed handler result types in my C
frontend; its previous Boolean-for-integer acceptance is corrected.

My C frontend and fresh self-hosted seed each pass seven executed source cases:
dynamic resume, lexical return, nested handlers, argument order and zero arguments,
scalar argument types, void operations and handler-local loops. Each also passes
ten rejection cases while preserving prior artifacts. Four metadata tests per
frontend retain sibling/nested parameter names, typed/zero-argument arms, lexical
return intervals, canonical round trips and execution after advisory stripping.
The self-hosted producer does not emit the C producer's DEBUG records; its test
still requires the complete local-binding metadata checks.

I rename opaque-value test bindings that used the already-reserved word handle;
I retain their original zero/nonzero and ownership assertions. I add a parse-success
assertion before the checker shadow indexes that fixture.

My eight-step compiler-product audit passes with all eighteen tracked production
inputs unchanged: self-compilation to NanoISA, verification, translation without a
VM wrapper, strict native compilation, CLI execution, paired effect controls,
handler metadata controls, and eight opaque/union/layout regression methods. The
regenerated compiler passes the same seven execution and ten rejection cases and
four metadata methods. These tests do not qualify installed Stage 1/2 or exact
release platforms.

I retain open implementation requirements: native nvm2c handler translation,
resource effect ownership, imported effect identity and full nominal/generic
signatures. A resource declaration currently causes the scalar handler control to
be refused for missing ownership lowering. Both frontends accept unused signatures
with unknown parameter/result types; my C frontend also accepts duplicate operation
names. Declaration controls retain those failures. An explicit return of a void perform
also remains refused by the self-hosted emitter, while the C-produced source
executes with result 7; my passing void statement case does not cover this form. I do not close #976 or full
effect support from the passing scalar cases.

GitHub API and Git HTTPS access currently fail from this environment. I retain the
coordination update body and API error; I have not published that update.
