# My self-hosted declaration-layout evidence

I continue #976 from 060309b99. I validate stored-value dependencies before self-hosted checking proceeds to emission. I follow record fields, tuple elements and concretely substituted generic union payloads. Arrays, Lists, HashMaps, borrow references and callable signatures stop the layout walk; unused generic parameters do not create value dependencies. Foreign record layouts remain trusted ABI declarations.

I track active and completed concrete type spellings separately for each root declaration, preserving template-parameter scope. An active type reached again establishes a cycle. I report cyclic by-value layouts with the declaration location. I separately refuse expansion beyond 128 nested generic substitutions; reaching that finite boundary is not proof of a cycle. Ordinary records and tuples do not consume that budget.

My initial old-compiler test records 14 accepted negative subcases. The checker component builds and executes its shadows, then passes six explicit public-entry-point cycle refusals and two acyclic controls. The first fresh seed passes 45 focused methods, including native/NanoISA output preservation, module-owner collisions and generic/callable regressions. Eighteen further methods pass explicit NanoISA emission, verified NanoVM execution and strict ASan/UBSan C AOT execution.

My unchanged-source eleven-step compiler-product audit passes in 539.158 seconds. Its native compiler product passes the four original layout methods. I then expose an overly broad depth limit with a 145-record acyclic chain. I restrict budget consumption to generic substitution; the corrected component accepts the same chain. The retained full audit precedes this final boundary refinement, and I keep that exact-source distinction visible.

The C frontend still accepts unused cyclic declarations through NanoISA emission (c-frontend-parity.log). Native C emission has its separate checked refusal, but frontend parity, aggregate union payload lowering and the complete 5.1 release/platform gates remain open.

My final corrected seed passes all 46 focused methods, including the 145-record chain, and all 18 explicit NanoISA/VM/C AOT methods. The corrected source-input hashes stay unchanged during the seed build. All cyclic and expanding-generic negative cases preserve prior native and NanoISA artifacts; imported equal-name record owners retain separate identities. The full eleven-step audit still refers to the checkpoint before the generic-only budget refinement, not the final exact release candidate.
