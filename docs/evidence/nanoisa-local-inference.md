# Ordinary NanoISA local inference

I tested source `91c5762ed4bcb6721deb38301d6c0a936d4842e5`, based on main `68bd6607bd88e0426d2e6cb0ea3621770d7d46d7` (through PR #565). PR #558 contains this bounded change.

I infer supported ordinary local types only when their annotation is absent. I reuse typed lowering, evaluate the initializer once, and retain explicit types and the annotation requirement for ambiguous empty constructors. My checked frontend now preserves concrete builtin `map_get` value types and declared enum-member identities, including lexical shadowing and key/member diagnostics. Fresh explicitly typed `array<int>` literals accept checked integer/enum elements; existing enum-array aliases remain invariant.

I validate local types before emitting an initializer and publish no binding after failed emission. I stop subsequent statements while preserving scope cleanup and the first diagnostic. These positive state and refusal checks do not establish the cause of the historical export-shadow incident `task_dd74b033c3984805bc27ce5017096c3c`; its preserved evidence and separate product acceptance remain open.

## Measured acceptance

I ran `make test-nanoisa-local-inference test-nanoisa-src-nano` on the source above. It exited 0 and completed a fresh native bootstrap, four canonical methods in 0.543 seconds, 86 baseline comparison checks, and 87 paired methods in 106.114 seconds. The complete log is `/tmp/nanolang-local-inference-restack.log`.

My unchanged `tests/nanoisa/fixtures/inferred_locals.nano` exercises scalar, enum, record, nonempty-array and returned-map locals, map access, lexical shadowing, mutation and once-only effects. C-seed/raw-selfhost comparison and canonical Stage1/Stage2 publication execute through VM and strict native compilation with output `once\n7\n`. Refusal controls preserve existing output for ambiguous constructors, unsupported types, bad keys, missing enum members, incompatible array aliases and mixed enum/bool literals. Shadow tests retain explicit unknown annotations, ordinary function identity and failed-lowering state.

Earlier evidence remains available: `/tmp/nanolang-local-inference-context-final.log` records the same complete implementation before restack (`ed319e28`), including fresh bootstrap, four canonical methods and 86 + 87 checks. Earlier canonical runs exposed missing map and enum inference; `/tmp/nanolang-local-inference-final.log` records the subsequent three compiler enum-literal context refusals. I repaired those prerequisites without reducing the full fixture. I do not classify the earlier failures as infrastructure.

## Bounded tasks

- `task_fd96d33b7d404449aab1e74b2b636628`: ordinary supported local inference.
- `task_0c5f2c64cc824d34a2c452111f18fb32`: concrete checked map access.
- `task_156884dbd35d476cbaea730ce3aa6152`: concrete checked enum members.
- `task_b7c4f9a51c894ad5aeb8f1adf5447219`: fresh enum literals under explicit integer-array context.
- `task_43128761d0b94470bcc13e700768454f`: defensive failed-lowering state, without historical incident attribution.

I retain unsupported-type refusals and the full-roadmap release hold. These tests establish the stated slice, not unrestricted inference or whole-compiler semantic equivalence.
