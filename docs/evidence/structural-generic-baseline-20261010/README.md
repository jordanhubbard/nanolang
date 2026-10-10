# My structural generic baseline

My C-seed-hosted self-hosted checker accepts the retained array<T> parameter
source, but my whole-module emitter returns `unsupported result type T` instead
of assembly. My generic detection currently recognizes a bare T parameter;
it does not infer T through array<T>. I require recursive structural binding
and substitution in both source producers, repeated-variable consistency and
executable shadow/program evidence. I preserve this baseline while my integrated
bootstrap source inputs remain frozen. This is an implementation gap under #976,
not a release qualification pass.

My C producer also refuses the source, treating T array elements as concrete
struct elements (cseed-baseline.log). Both producer paths require implementation.

## My staged matcher

I prepare binding-candidate.nano.txt outside my frozen bootstrap inputs. It
extends the existing generic type-variable classifier with whole-type matching
for nested constructors and callable spellings, balanced concrete-type capture,
repeated-variable consistency, and copied prior bindings. Its mandatory shadows
pass and the C seed compiles its native driver successfully. The driver main is
trivial; I have not yet established native runtime matching or connected the
helper to either production checker/emitter. I retain the ownership and shared
array alias failures caught while developing it. Typed local lengths avoid the
legacy native strlen signedness failure retained separately.

I next need to integrate generic detection, binding and substitution together:
the checker must infer structural variables and return concrete annotations;
the emitter must specialize every parameter/result and local occurrence; C
TypeInfo inference must match those semantics. Source programs, imported calls,
negative identity/shape checks and installed-stage qualification remain required.
