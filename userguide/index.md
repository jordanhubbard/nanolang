# NanoLang User Guide

I am NanoLang. I use explicit boundaries, prefix calls, equal-precedence operators, and tests beside the functions they exercise.

```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    }
    return (* n (factorial (- n 1)))
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
}
```

## Start Here

1. [Build me and run a first program](guide/01_getting_started.md).
2. [Learn my language](guide/02_language.md).
3. [Use structured data and typed errors](guide/03_data_and_errors.md).
4. [Work with modules, foreign code, and resources](guide/04_modules_and_ffi.md).
5. [Understand shadows, tests, and my verified boundary](guide/05_testing_and_trust.md).
6. [Choose a tool or backend](guide/06_tools_and_backends.md).
7. [Measure native performance and tune from evidence](guide/07_performance_profiling.md).

## Reference

- [Examples](generated/examples.md) are generated from every `.nano` file under `examples/`.
- [Builtins](generated/builtins.md) come from the mechanically checked standard-library reference.
- [Modules](generated/modules.md) are generated from the module tree and manifests.
- [Compiler CLI](generated/cli.md) comes from the compiler used to build this guide.
- [NanoISA](https://github.com/jordanhubbard/nanolang/blob/main/docs/NANOISA.md) is my shared typed VM boundary.

## What I Promise

- Function calls use `(function argument)` syntax.
- Function parameters and return types are explicit.
- Local bindings may use inference when the initializer makes the type plain.
- All infix operators have equal precedence and associate left to right.
- Project policy requires useful shadows for changed named functions. Compiler enforcement has documented exemptions.
- A passing shadow is a test, not a proof.
- My C backend is the production compilation path. Other backends have narrower documented subsets.

The parser, typechecker, builtin registry, and tests are the authority when an old document disagrees. I would rather correct a guide than preserve a confident mistake.
