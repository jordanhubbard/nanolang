# I preserve prepared mutation roots across later failures

I qualify fixture checkpoint `14fb2fa43d19d89dde870c56f9708b33c6d3cd51`
on merged842 production. My [contract](../NANOISA_OWNED_ARRAY_MUTATION_RUNTIME.md)
and [seal](owned-array-mutation-runtime.json) identify exact inputs and all logs,
artifacts, prepared providers, compiler executables and per-phase maps.

| Fresh phase | Seconds | Result |
| --- | ---: | --- |
| GCC ordinary mutation | 7.344 | PASS |
| GCC ASan/UBSan mutation | 23.593 | PASS |
| Clang18 ASan/UBSan mutation | 22.926 | PASS |
| GCC unchanged private corpus | 10.412 | PASS |

Each mutation configuration runs true switch and computed-goto with explicit
preprocessor evidence, both fusion settings, four synchronous public APIs and
native O0/O2. Each VM binary reports1906 checks. The unchanged adjacent corpus
reports4746 checks per dispatch binary. I preserve normal output `1\n7\n1\n`
and failure prefix `1\n`, exact status/result conventions, full root/byte cleanup
and fresh-VM same-process recovery using the same admitted module.

My call fault intercepts the real eight-to-at-least-thirteen-slot realloc before
callee activation, with the prepared ARRAY receiver below the moved Handle.
My VM pack faults identify the decoded OWN_PACK next-PC and actual shell/field
allocation sizes after the append marker. Native's one shell/inline-field
allocation uses its exact size; fixed native helper arrays introduce no fictional
call-preflight allocation. Existing growth/accounting controls remain separately
qualified and covered by the unchanged corpus, not relabeled as new fault sites.

I instrument fixture, VM, heap, native emitter and generated native translation
units in sanitizer phases; copied linked provider objects remain ordinary.
I do not claim whole-provider sanitizer coverage. I copied162 already qualified
providers/CLIs with exact hashes and inventoried two compiler executables.
All164 hashes match around each phase and at the end; all1768 inventoried source
files match before/after. This does not inventory every transitive system tool.
The600-second external bound returned0. All four first fresh phases passed.

I seal488 files and12 current production/fixture inputs. I change no runtime or
producer behavior. Task5652 remains pending canonical review/merge; parent430220,
source18731 and mutation bba622 remain separate acceptance obligations. This Linux
runtime evidence makes no Darwin or paired source completion claim.
