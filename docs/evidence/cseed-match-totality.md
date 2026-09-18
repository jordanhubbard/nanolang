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

## Focused immutability seal

I separately rebuilt only `bin/nanoc_c` in a fresh detached checkout at
`e6699c22fc51f4d7ee75413db184a6fc49d56896`, then ran the same seven focused
methods. All seven passed in 1.588 seconds. This is a focused-only seal; I did
not rerun or relabel the bootstrap evidence above.

Before and after the focused run, the checkout remained clean at tree
`399d1a1803906a886456348595901392a9d54238`. Every file tracked by this pull
request had the same SHA-256 before and after:

```text
fd212b21a370dde1cf1833f5a685b6b9cc0601e6906ace09a702c184919aafd5  docs/ROADMAP.md
d9dca642f9fbfce1dae7878560531b7c37fd921c1bd8f2a6f3cdd0c7761727c8  docs/evidence/cseed-match-totality.md
f705a6e706ae5554ce3f56b1697ebdcdac3e7a49b49773954c45c315a367634b  src/nanolang.h
3676713df00069c45f1957aba4b97f8a7636296b3fe54e66bbab85ac7706d54e  src/parser.c
51265923b556ed682ec7d643abe040c069d98128c2bb2abaa30b430d37461543  src/transpiler_iterative_v3_twopass.c
c55ef5b0a0c15730973f82a36bb7165e8bb21677bacfadde6f695f2ee4ff3db3  src/typechecker.c
f275e6c625622880b28a987441e649328af324d9448184c3aa1dbc43740e42ed  tests/test_cseed_match_totality.py
```

The exact C-seed compiler and resolved tool executables were also unchanged:

```text
5d55eeb5685a9789318a9542dd58667717cdc14447f46a66ec0e7f52b61fc4eb  bin/nanoc_c
1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9  /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang
07919b0fbf82742c24420d7dcadbd459f61bc1feb638eca02b81dccab2daf4d5  /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14
b8763cf250e607a778bb4603cecb5b90338814d0a3dfcba0d57b1de242f610e9  /usr/bin/make
b8763cf250e607a778bb4603cecb5b90338814d0a3dfcba0d57b1de242f610e9  /usr/bin/git
b8763cf250e607a778bb4603cecb5b90338814d0a3dfcba0d57b1de242f610e9  /usr/bin/cc
178e19cfe796ce33815195c42d34562c20ec4d5131534fb7e6a55b91f14371d5  /usr/bin/xcrun
0812595f981a26f813d98dc380af14d4af427626c9339eda29eb849ae13de1e3  /usr/bin/shasum
```

`/usr/bin/cc` is an Apple tool dispatcher. The separately resolved and hashed
compiler above is the actual Apple Clang 21.0.0 executable selected by
`xcrun --find clang`; I do not treat the dispatcher hash as the compiler hash.
The retained focused seal is:

```text
64509dd8276e68f3693ad2e83ba8132f04ebc2d44ecfcdb3b32b046ac1181d8f  /private/tmp/nanolang-match-totality-seal.LgtkMY/evidence/focused-before.txt
bca750d735ae834440751f128e0518ce8ea5cca7dc0c6a1ab2b8cf28b20c7e27  /private/tmp/nanolang-match-totality-seal.LgtkMY/evidence/focused-run.log
f0091ab3d708da48d936d56a1fd2dd3dc49f520c3f8d19e7998ec52473da2c06  /private/tmp/nanolang-match-totality-seal.LgtkMY/evidence/focused-after.txt
05c74651bf8f8650dc2f7751c74f6e25f017a667e08bc5cf9db44d8518964a06  /private/tmp/nanolang-match-totality-seal.LgtkMY/evidence/source-hashes.before
05c74651bf8f8650dc2f7751c74f6e25f017a667e08bc5cf9db44d8518964a06  /private/tmp/nanolang-match-totality-seal.LgtkMY/evidence/source-hashes.after
cab1aa8608082928ebacf312e5c0fa5445d36879de46513125c4def13b3c9330  /private/tmp/nanolang-match-totality-seal.LgtkMY/evidence/tool-hashes.before
cab1aa8608082928ebacf312e5c0fa5445d36879de46513125c4def13b3c9330  /private/tmp/nanolang-match-totality-seal.LgtkMY/evidence/tool-hashes.after
```

I used Apple Clang 21.0.0, Python 3.14.6 and GNU Make 3.81. This evidence is
bounded to the C seed. It does not align the interpreter, NanoVirt, the
self-hosted checker or public C wildcard dispatch, and it does not close the
shared no-success or wildcard-ordering obligations.
