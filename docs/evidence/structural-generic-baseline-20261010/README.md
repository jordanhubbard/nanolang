# My structural generic baseline

My C-seed-hosted self-hosted checker accepts the retained array<T> parameter
source, but my whole-module emitter returns `unsupported result type T` instead
of assembly. My generic detection currently recognizes a bare T parameter;
it does not infer T through array<T>. I require recursive structural binding
and substitution in both source producers, repeated-variable consistency and
executable shadow/program evidence. I preserve this baseline while my integrated
bootstrap source inputs remain frozen. This is an implementation gap under #976,
not a release qualification pass.
