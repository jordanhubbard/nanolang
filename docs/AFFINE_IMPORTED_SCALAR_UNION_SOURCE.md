# I retain imported generic-union identity through source emission

I extend `task_a18a9f752536469faafc4d3ebec01dfd` through the bounded
child `task_796dec1897c14b66b6b7e4b03fce1acd`. My first semantic source
checkpoint proves two concrete instances declared in the root file. It does
not establish that a dependency-owned declaration survives module loading,
alias resolution, selected shadows and independent source emission.

## My accepted dependency closure

One dependency declares `Choice<T,U>` and public functions using
`Choice<int,string>` and `Choice<float,bool>`. The root imports that dependency
under a named alias and observes both concrete instances through constructors,
parameters, results, statement matches and value matches. The dependency and
root shadows inspect exact payloads; an `assert true` shadow is not evidence.

The same physical dependency may be imported through two aliases. Both aliases
refer to one declaration identity. A different dependency may declare the same
short generic-union name, but its concrete values are not interchangeable.
Declaration owner plus recursive concrete type arguments remain authoritative;
the retained display spelling is only a wire name and diagnostic aid.

I admit scalar payloads already accepted by the import-free profile. Resource
payloads, nested aggregate payloads, wildcard expansion, public
`ARRAY_FIELDS`, and arbitrary imported graphs remain separate work.

## My independent producers

I compile the unchanged root and dependency with:

- C-seed `nano_virt --emit-nvm`;
- installed Stage 1 `--emit-nvm`; and
- installed Stage 2 `--emit-nvm`.

NanoVirt exercises the C source producer. Stage 1 and Stage 2 exercise the
self-hosted Nano source producer built from separate compiler stages. Every
route selects dependency shadows by default, verifies the resulting module,
executes it in NanoVM, translates it through nvm2c, compiles strict sanitized
native C and produces the same exact output.

The raw `nanoisa_emit` API accepts one already-bound parser and deliberately
does not discover imports. I keep its complete import-free parity gate from the
preceding checkpoint; I do not relabel a manually concatenated source file as
dependency loading. The canonical drivers own imported-program assembly.

## My refusal boundary

Every refusal preserves prior output and reports its actual phase. I cover:

- a value from a separate same-spelled dependency passed to the first module;
- an incompatible concrete instance crossing the imported call boundary;
- a wrong substituted payload type;
- an unresolved or ambiguous imported declaration identity; and
- a failing dependency shadow preventing root publication.

I preserve the first terminal before each correction. I do not weaken module
visibility, flatten two declaration owners into one name, bypass selected
shadows, or admit kind 2 to make this slice pass.

## My acceptance boundary

Fresh Linux and Darwin qualification must record source, compiler and runtime
identities. This slice does not close mixed `UNION_VARIANTS` plus
`ARRAY_FIELDS`, resource-bearing union payloads, the full affine parent,
PR522, product acceptance or release publication.
