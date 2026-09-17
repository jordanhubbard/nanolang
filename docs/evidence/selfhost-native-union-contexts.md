# I preserve native union payload contexts and globals

My self-hosted native stages previously emitted `nl_Box_T` for the payload
of `Envelope<Plain>` and initialized an aggregate union global with scalar
`0`. I reproduced both failures with freshly bootstrapped compilers.

I now apply my shared type-parameter substitution to the complete payload
annotation before lowering it to a C type. This preserves nested generic
arguments and does not replace a parameter name inside another identifier.

I declare union globals first and evaluate their initializers through my
existing expression emitter in the startup constructor. I collect those
assignments and guarded scalar initialization in declaration order. The
initializer environment retains the preceding globals' types and guarded
access rules; function prototypes precede the constructor. Ordinary literal
and other existing global declarations retain their previous behavior.

My executable tests cover concrete record-array payloads in globals, calls,
returns, assignments and nested unions; scalar nested union values; and
ordered effectful global factory calls with guarded scalar state and a copied
union global. Helper shadows check nested type spelling and actual aggregate
initialization output.

## Boundaries I found and kept open

`task_98dab2489ab749e7b9053f944b3fe489` tracks specialization discovery when an
inner concrete union appears only through an outer payload. My current
annotation scan does not emit `Inner<int>` in this example unless another
annotation names it. `task_7bc727794375435ca72e6fef466ff161` separately tracks
the direct projected nested match, which currently emits `nl_UNKNOWN`:

```nano
union Inner<T> { Value { value: T } }
union Outer<T> { Value { value: Inner<T> } }
fn main() -> int {
    let outer: Outer<int> = Outer.Value { value: Inner.Value { value: 42 } }
    match outer {
        Value(payload) => {
            match payload.value { Value(inner) => { assert (== inner.value 42) } }
        }
    }
    return 0
}
shadow main { assert (== (main) 0) }
```

I retain the source in `/tmp/nanolang-native-payload-inference.nano`, baseline
failure logs in `/tmp/nanolang-selfhost-union-contexts-baseline.log`, and the
post-substitution failure in `/tmp/nanolang-selfhost-union-contexts-first.log`.
My passing nested-value control uses an explicitly typed inner reader; it
does not establish either missing behavior above. Resource-collection
ownership and full compiler bytecode bootstrap remain separate work.

## Validation

Fresh `make bootstrap` passes. All fourteen methods in
`python3 -m unittest -v tests.test_selfhost_generic_contexts tests.test_native_nominal_order tests.test_native_nested_generics`
pass in 100.684 seconds (33 compiler decisions, including the existing
expected layout-cycle refusals). The new target participates in `test-units`.

I also rebuilt compiler bytecode with the C frontend, translated and compiled
it to a native compiler, and ran help/default-native hello plus explicit
`--emit-nvm` hello. That same hello module passes NanoVM and native AOT
execution. Commands and products are retained in
`/tmp/nanolang-union-contexts-product`; the compiler host-C build took 75.078s.
This checks regression of the bounded bridge, not self-emission or a bytecode
fixed point. Bootstrap and focused logs are
`/tmp/nanolang-selfhost-union-contexts-final-bootstrap.log` and
`/tmp/nanolang-selfhost-union-contexts-final.log`.
