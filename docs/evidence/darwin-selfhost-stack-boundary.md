# My Darwin self-host stack-boundary evidence

I qualify the bounded compiler-product policy in
[`DARWIN_SELFHOST_STACK_BOUNDARY.md`](../DARWIN_SELFHOST_STACK_BOUNDARY.md)
for `task_e1598bc864212c669cc8b4cb2b73be55`. I do not use this evidence as a
fixed-point, scalar-union, full-product or release claim.

## My source and tools

I started from clean canonical main
`209a4d5588835902b1ac30dd40b42e761ab7b994`. My production source at the
passing gate hashes to:

```text
4096cf86fbe78a0017fae26f7743cc515b27460ed51e05c07b609a6cec098741  src_nano/nanoc_v06.nano
```

I ran on macOS 26.6.2 arm64 with GNU Make 3.81 and Python 3.14.6. The actual
compiler selected through Xcode was Apple Clang 21.0.0 at
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`,
SHA-256
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.

## My fresh bootstrap

I ran:

```text
make -j8 bootstrap
```

C-seed, Stage 1, Stage 2, the installed hello smoke and the no-C-seed smoke
passed. The raw log is
`/private/tmp/nanolang-e159-current-bootstrap.log`, SHA-256
`85dcc620d055fe5650ace6e5e59720aad0ffc7bcc2e8e3564cc4e5bdef45cd8f`.
The resulting selected compiler hashes were:

```text
162e82e73462bb948edfadc0c9202dc73b7a2217c2631b216be76f907035f55e  bin/nanoc_c
b4f788f63af0d8180f3614edaef30bd4b0c6c413dc1f9364bdf41562a4cd0943  bin/nanoc_stage1
3cff6a00df19edd1b2a9a6404bda2d6377190daf55ed4161b795df0e6a5b494a  bin/nanoc_stage2
3cff6a00df19edd1b2a9a6404bda2d6377190daf55ed4161b795df0e6a5b494a  bin/nanoc
```

The native Stage 1 and Stage 2 binaries differ. The bootstrap reports that
fact, and I do not describe this result as a native fixed point.

## My policy controls

With `NANO_CFLAGS` absent, freshly built Stage 1 compiled unchanged hello and
printed a native command containing `-O1`. An explicitly empty value follows
the same policy because the environment interface supplies the same empty
string. The produced program ran and printed `Hello from NanoLang!`. The raw
log is
`/private/tmp/nanolang-e159-default.log`, SHA-256
`5399ea4768b2895a7980046eea400605f5047cabe26fbc3f077c6b0e06475836`.

With `NANO_CFLAGS='-O0 -g'`, freshly built Stage 2 compiled the same hello and
printed a native command containing that exact setting and no injected
`-O1`. The produced program also ran and printed `Hello from NanoLang!`. The
raw log is `/private/tmp/nanolang-e159-explicit.log`, SHA-256
`db7e06fb5391d46666185237b0565a6cb6c887895133103525c5cc6a61cc9f2f`.

The helper shadow covers the empty/default setting, an explicit debug setting
and an explicit sanitizer setting during both self-hosted bootstrap stages.
These controls qualify only the empty/default-versus-nonempty flag boundary;
they do not claim that absence and an explicitly empty environment value are
distinguishable. The retained frame measurements and rejected heap-pool
hypothesis remain in the bounded contract; a generated-C ABI redesign remains
separate work.
