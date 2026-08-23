# Testing and Trust

I distinguish tests, static checks, and proofs. They answer different questions.

## Shadows

A shadow is an executable test attached to a function:

```nano
fn double(value: int) -> int {
    return (* value 2)
}

shadow double {
    assert (== (double 0) 0)
    assert (== (double 3) 6)
}
```

During ordinary compilation I execute shadows in the host interpreter. Generated native programs also contain their shadow harness. A shadow tests the cases it executes; it does not prove the function for every input.

Compiler enforcement and project policy differ:

- The compiler normally warns about a missing shadow.
- It exempts extern functions, `main`, generated lambdas, GPU functions, and functions that call externs.
- Repository policy requires a useful shadow for every added or changed non-extern named function when it can be tested.

## Property Tests and Coverage

Property tests sample generated values and can shrink failures. Sampling is not exhaustive verification. Coverage reports which code executed; it does not establish correctness.

## Contracts

`requires` checks preconditions and `ensures` checks postconditions at their implemented boundary. A successful runtime check says that condition held for that execution.

## Formal Verification

My Coq development proves stated metatheory for a defined core model. It does not automatically prove every backend, foreign call, allocator, module, or user function.

Use the trust report to inspect the boundary:

```bash
./bin/nanoc program.nano --trust-report
```

Say **proved** only for a checked theorem in its stated model, **tested** only for behavior exercised by a named test, and **assumed** for unchecked platform or foreign behavior.

## Useful Checks

```bash
make test
make userguide-check
make check-stdlib-docs
python3 scripts/check_markdown_links.py
```
