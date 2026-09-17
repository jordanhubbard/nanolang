# Resource callback signature boundary

I reject generic resource-bearing callback signatures until their parameter and
result transfer semantics are verified. This is MAC
`task_f05070eff34a4b169318fe949da45630`, PR #489, beneath the open first-class
signature/ownership acceptance work.

## Behavior

I inspect retained C FunctionSignature TypeInfo for callback parameters,
results and nested returned callbacks. My matching self-hosted resource pass
walks concrete annotation strings. Explicit/inferred locals, forwarded and
returned callbacks, and explicit global callback signatures retain the same
boundary. Rejection occurs before artifact publication and says that generic
resource callback signatures need ownership lowering.

I preserve nongeneric resource callbacks. My tests execute a `fn()->Handle`
factory and a `fn(Handle)->int` consumer with owned destructuring. `Box<int>`
and phantom `Marker<Handle>` callbacks remain ordinary controls. A callback
value is not itself a resource owner: its signature is a separate contract.
My previous self-hosted classifier confused `fn()->Marker<Handle>` with a live
resource. That failure also reproduces with the earlier validated Stage 2.
The classifier now treats function values separately, with mandatory helper
shadows covering this distinction.

Before this guard, my C frontend accepted the generic resource callback
annotation under source-only C emission; the native path then failed on its
unsupported function typedef. I do not describe that earlier result as working
native callback support.

## Validation

At `51468171`, I passed a fresh normal-budget bootstrap and 78 paired methods
across these suites (176.343 seconds):

- `tests.test_resource_callback_boundary`
- `tests.test_generic_function_values`
- `tests.test_affine_contract_boundaries`
- `tests.test_generic_selected_ownership`
- `tests.test_global_resource_boundary`

The complete C typechecker and resource-classification suites also passed.
The first bootstrap attempt exposed generated-list Parameter initialization
missing from the merged module metadata path. PR #490 independently repaired
that initialization; I retained the original failure log and backtrace.

At integrated checkpoint `1b84a622`, after rebasing onto merged PR #490 and
removing the temporary equivalent repair commit, all 13 boundary methods passed
again (32.433 seconds). Relevant C and self-hosted compiler inputs and these
tests are identical to tested `51468171`; the upstream source difference is an
unrelated nvm2c correction.

The dedicated gate is `make test-resource-callback-boundary`. Its nine negative
cases require the ownership diagnostic and preservation of an existing artifact;
four positive cases compile and execute their mandatory shadows and native
program through all three stages. Local evidence is retained under
`/tmp/nanolang-resource-callback-`: `bootstrap.log` and `backtrace-native.log`
retain the initial module failure; `bootstrap-r2.log`, `paired.log`, `cunits.log`
and `restack.log` record the passing gates. The earlier phantom failure is in
`phantom-baseline.log` and `selfhost-check.log`.

## Boundary

I do not implement generic resource callback transfer or declare complete
first-class ownership conformance. Resource collections remain unsupported.
Ordinary callsite type compatibility, imported signature serialization, full
ownership metadata verification and the remaining affine contract have their
own gates. This tested guard does not close those obligations.
