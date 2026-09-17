# My compiler artifact import boundary

I retain an imported declaration's source owner separately from its C-linkage
name. My canonical binder supplies those owners and source paths. My emitter
uses the declaration's manifest-backed immutable build generation as the
library namespace, writes the exact parameter/result signature, and marks the
import `artifact`. I do not turn path helpers into empty-namespace builtins.

I support string-result artifact calls for `path_basename`, `path_dirname`,
`path_normalize`, `path_canonical`, `path_join`, `path_relpath`, and the internal
`nlc_module_artifact` compiler host operation. Their supported arities and string
parameters are explicit. I reject mismatched signatures, operand types, missing
source ownership, and unavailable builds. My raw-source API clears ambient
binding state; it cannot infer a foreign provider from a same-spelling symbol.

My compiler host facade invokes the existing module builder. It returns the
library inside the generation reported by that build, without rereading a
mutable `current` selector. It does not compile NanoLang expressions or run a
VM. My NanoLang wrapper snapshots the C helper's borrowed result before another
call can replace it.

I tested:

- Three facade methods: stable cache reuse, changed-source generation identity,
  retained old bytes/results, rejected builds and invalid inputs. A retained
  language string survives a subsequent call that clears the C helper result.
- Two libraries exporting the same C symbol: qualified and selected aliases
  preserve owner identity through assembly, verification, VM and native runs.
- Five exact C-seed/self-hosted checks for bytecode plus ordered library names,
  import kinds, parameter types and results.
- Rejection of malformed signatures, operand counts/types, unbound calls and
  missing source paths; bound emission followed by raw emission resets state.
- A real canonical filesystem program with basename, dirname, join and
  normalization: repeated bytecode matches and both execution backends pass.

The native filesystem check links my existing `nano_aot_runtime.o` host ABI and
exports it to foreign libraries. Omitting that documented link contract caused
an initial `gc_release` loader failure in this new test; using the supported
contract passes without runtime changes. My custom scalar-only artifact fixture
needs no host runtime.

My actual compiler emission advances past the path imports and stops at
`unsupported extern result or symbol nl_nanoisa_assemble_text_save`. That
int-result artifact contract is the next recorded slice
(`task_3560b1472ae64d73b0449b7df6933914`). This slice does not establish a NanoISA
bootstrap or a fixed point.
