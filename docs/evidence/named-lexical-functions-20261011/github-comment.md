I complete named lexical-function parsing, scoped checking, source binding names and recursive capture retention through both NanoISA frontends, NanoVM and native C AOT under #976.

My final audit passes all seven compiler-product steps with fourteen production inputs unchanged. Both C and the fresh self-hosted compiler pass 21 source/metadata methods; the product also passes 22 nominal/layout methods. VM, translator, bytecode, environment/typechecker and changed-code sanitizer gates pass. Evidence: docs/evidence/named-lexical-functions-20261011/README.md.

I correct my release prioritization against RELEASE_5.1_SCOPE.md and NANOISA_ONLY.md: the retained AST-to-C nested-function failure is a legacy compatibility follow-up, not a new One IR release gate. Full producer-name coverage, frontend facts, reconstruction, profiles, remaining service/translator work and exact release-platform qualification remain open. I keep #976 open.
