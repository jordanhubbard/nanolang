# I keep global initializers in their declaring module

I reproduce all four failures from Linux qualification39205176925: same-named
int/string globals in two modules caused my first module's function to see the
second module's string. The initializer pass ignored the returned lexical-let
symbol array, but array_push mutated its shared input. Unqualified names then
won lexical lookup in every function.

I publish initializer symbols under the same mb_lookup owner/name identity used
by global declaration registration. Ordinary lexical lets retain their names;
source AST names and diagnostics remain unchanged. The shared checker handles
inferred/declared/destructured symbol publication consistently. I do not expand
global type inference or ownership admission.

My changed C-hosted Nano compiler runs all dependency/root shadows and passes
all29 imported-global methods in9.666s, including the original four failures,
module aliases, mutation, captured locals, dependency shadows and refusal output
preservation. New controls cover global initializers referring to their own
module, parameter/local shadowing and both module orders. The C producer passes
the new explicitly typed fixture. My first fixture used unannotated public globals,
which the C producer rejected; I retain that setup failure and correct its declared
types without weakening module-context assertions.

I retain source/provider hashes and complete build/test logs. This is a focused
changed C-hosted producer receipt, not fresh Stage1/Stage2 or Linux/Darwin proof.
Issue986 remains open until those gates pass. Review found no blocking issue in
the named-publication boundary.
