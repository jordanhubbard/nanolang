# My C-seed match totality boundary

I use one checked domain and coverage decision for value and statement matches.
The checker accepts exact `int` matches with an unconditional wildcard and
exact known-union matches with unconditional finite coverage. It refuses mixed
arm families, wrong scalar domains, unresolved union identity, non-`bool`
guards and conditional-only coverage before publication.

I retain the accepted domain in the match AST after typechecking. Both
iterative C lowering paths consume that fact. A wildcard-only integer match no
longer depends on finding an `INT:` arm, and a missing checked union identity is
a compiler invariant failure rather than fabricated `0` or an empty statement.
The scrutinee is still evaluated once into `_m`.

## Fresh bounded qualification

I qualified detached source
`eba4d2b8271cd4dbd569ce3821c907238995453a` on Darwin. This is production
checkpoint `19d25bd751abcdbb5c5854e9c45964d9bd37e9cc` plus the reviewed test-only
checkpoint.

- `make -j8 bootstrap` passed in 304.96 seconds. Stage 1, Stage 2, both hello
  smokes, the installed compiler smoke and the no-C-seed check passed. The
  native Stage 1 and Stage 2 binaries differed; the gate records that expected
  distinction and does not claim a native fixed point.
- `python3 -m unittest -v tests.test_cseed_match_totality` passed all seven
  methods in 1.51 seconds. The controls execute complete union and wildcard-only
  integer matches in value and statement positions, check once-only integer
  scrutinee evaluation, and retain prior output for every checked refusal.
- The refusal controls cover incomplete and conditional-only union coverage,
  missing integer wildcard coverage, non-`bool` guards, wrong scalar domains,
  mixed and cross-family patterns, and unresolved union identity.

The source and test files were unchanged after qualification:

```text
c55ef5b0a0c15730973f82a36bb7165e8bb21677bacfadde6f695f2ee4ff3db3  src/typechecker.c
51265923b556ed682ec7d643abe040c069d98128c2bb2abaa30b430d37461543  src/transpiler_iterative_v3_twopass.c
f275e6c625622880b28a987441e649328af324d9448184c3aa1dbc43740e42ed  tests/test_cseed_match_totality.py
```

The retained raw logs are:

```text
181813ac7c310f6c60fe8f29257197666cef7dffb03db0e5861b29ec68f486a9  /private/tmp/nanolang-match-totality-qual.WywuHg/bootstrap.log
96508f3c37b2f2161a8c18f8831079836e8dfcfbefca58fb4d732a44345d47f6  /private/tmp/nanolang-match-totality-qual.WywuHg/cseed-match-totality.log
```

I used Apple Clang 21.0.0, Python 3.14.6 and GNU Make 3.81. This evidence is
bounded to the C seed. It does not align the interpreter, NanoVirt, the
self-hosted checker or public C wildcard dispatch, and it does not close the
shared no-success or wildcard-ordering obligations.
