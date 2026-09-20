# My Darwin self-host stack-boundary evidence

I qualify the bounded compiler-product policy in
[`DARWIN_SELFHOST_STACK_BOUNDARY.md`](../DARWIN_SELFHOST_STACK_BOUNDARY.md)
for `task_e1598bc864212c669cc8b4cb2b73be55`. I do not use this evidence as a
fixed-point, scalar-union, full-product or release claim.

## My source and tools

I integrated the bounded production change with canonical main
`8ed0a0ff6e30969b9721aa0cf89db011ec4f4506` and qualified clean detached head
`53b56b62dea1b8374ca6e42866d762e7d47b714c`. My production source at the
passing gate hashes to:

```text
7003aa6c526cb73a3c8da6df81185c2af7f41955113c65e055dacaf8f93ec740  src_nano/nanoc_v06.nano
```

I ran on macOS 26.6.2 arm64 with GNU Make 3.81 and Python 3.14.6. The actual
compiler selected through Xcode was Apple Clang 21.0.0 at
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`,
SHA-256
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.

## My fresh bootstrap

My first sparse-checkout attempt stopped before compilation because I had not
included tracked prerequisite `spec/nanoisa.yaml`. I retain the setup terminal
at `/private/tmp/nanolang-pr892-current-evidence.BIRMgz/bootstrap.log`, SHA-256
`fb2d55e0e5e8df9d30403a05dcf3aaaf748f89825dd9eebb77a819d01a75bc9b`.
There was no compiler artifact to replay. I added the complete tracked `spec/`
directory to the immutable checkout and ran:

```text
make -j8 bootstrap
```

C-seed, Stage 1, Stage 2, the installed hello smoke and the no-C-seed smoke
passed in 361 seconds. The raw log is
`/private/tmp/nanolang-pr892-current-evidence.BIRMgz/bootstrap.corrected.log`,
SHA-256
`c6cf1072536c49791c55311a103c61052fff8fd75559108caa11c194bff57803`.
The resulting selected compiler hashes were:

```text
6cb355dffb0dc7f5e616d41eeeb335216ff7eca720bcb1ef8c983c9353a7eea1  bin/nanoc_c
80f89b5c33bc083c0578ecb18bacd436aa9e30f6d8505eed2a3d2d5bb322cc02  bin/nanoc_stage1
5717b48a84bd96ecad7857e412b89c98d304fb2cfd9c95064f34e0c2bd1b0a44  bin/nanoc_stage2
5717b48a84bd96ecad7857e412b89c98d304fb2cfd9c95064f34e0c2bd1b0a44  bin/nanoc
```

The native Stage 1 and Stage 2 binaries differ. The bootstrap reports that
fact, and I do not describe this result as a native fixed point.

## My policy controls

With `NANO_CFLAGS` absent, freshly built Stage 1 compiled unchanged hello and
printed a native command containing `-O1`. An explicitly empty value follows
the same policy because the environment interface supplies the same empty
string. I tested both cases separately. Both produced programs ran and printed
`Hello from NanoLang!`. The absent log SHA-256 is
`afaf6ce344aaa8e853674c463bd3b3a668b9d03800a170e4712dd29efe3d5946f`;
the explicitly empty log SHA-256 is
`5e116f7442ff698d6d34e5e6412825700cb42c45c5c3d0380bae08e030867a9f`.

With `NANO_CFLAGS='-O0 -g'`, freshly built Stage 2 compiled the same hello and
printed a native command containing that exact setting and no injected
`-O1`. The produced program also ran and printed `Hello from NanoLang!`. The
raw log SHA-256 is
`9d30db1c89cb81e42979e1cfd56d422100ada115ad0e9cf1f78b611476114a6f`.

The helper shadow covers the empty/default setting, an explicit debug setting
and an explicit sanitizer setting during both self-hosted bootstrap stages.
These controls qualify only the empty/default-versus-nonempty flag boundary;
they do not claim that absence and an explicitly empty environment value are
distinguishable. The retained frame measurements and rejected heap-pool
hypothesis remain in the bounded contract; a generated-C ABI redesign remains
separate work.

The 57,284-entry tracked index, 3,103 checked source files and selected host
tools are byte-identical before and after the gate. The sealed evidence archive
is `/private/tmp/nanolang-pr892-current-evidence.BIRMgz.tar.gz`, SHA-256
`9417073db58df7df44e2de2dbaa04cbc70c9c7658d935fecf318fbd0813ea1b7`.
