# My native module link order

My artifact-support module exposed two failures during a fresh three-stage
bootstrap: its manifest listed the same canonical cJSON source my native driver
already supplies, and my driver placed pkg-config libraries before the module
objects that used them. The linker reported duplicate cJSON definitions and an
unresolved SHA256 call.

I now collect module objects before their library flags. I compare canonical
source paths against my already supplied cJSON source; a different file with
the same basename remains part of the link. I retain ordinary self-contained
module manifests. I do not disable duplicate-definition or unresolved-symbol
checks.

`tests/test_native_module_linking.py` builds and runs two real native programs.
Both invoke SHA256 from an imported C module. One also uses a manifest entry
for my existing cJSON source; the other supplies its own different `cJSON.c`.
Both fail with the previous compiler and pass with a freshly built standalone
Stage 1 compiler. The combined artifact-import and link repair also passes a
fresh three-stage native bootstrap and the installed-compiler check without the
C seed. These fixtures live inside the
repository, matching the compiler bootstrap's source context.

A separate initial fixture outside the repository exposed omitted native
module metadata when source-ancestry root discovery returns empty. I recorded
that boundary as `task_c6b698326e0f4e6296299ddfdf172ebd`; this source/order repair
does not claim to fix it.
