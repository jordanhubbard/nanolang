# Self-hosted native `u8`

I qualified the bounded scalar native-C repair at production commit
`cfc6f3b93c4812deca1a0244b933ea19599f0d9a` and focused test commit
`728a38eb3a6c82bd474e6326fe0c82e1ca66583d` on Darwin arm64.

## Preserved failures

The original MAC worker committed `cf0e0f587c45540c9d972158e7db08f04a4dd881`
but could not publish it. Its finalizer stopped while building
`src/interpreter_ffi.c` because that Linux OpenShell image had no `ffi.h`.
I retain that terminal as MAC evidence `ev_4ef674d7a08e4d42aa2f0c4eda4dc058`;
the worker stderr has SHA-256
`d9d0d9c999a56fbcc09ee43292a5f47bea159a19d732ea11a3b5662f1a223dec`.
It is recovery input, not acceptance evidence.

Before rebuilding the recovered source, the installed Stage 1 compiler refused
the unchanged focused fixture with unknown `nl_u8` declarations and an
unresolved `nl_cast_bool`. The complete diagnostic is retained at
`/private/tmp/nano-u8-stage1-before.Awa5GQ/stdout.log`, SHA-256
`93edd2ebbb7fbb8cc9fc288c931b451e8ad5881538bf47347bddcfe76d4a57c0`.
The pre-existing output stayed byte-identical to `previous accepted output`,
SHA-256 `281e67610596514d338ef6881995935a5bd72e5f42c8f706d354bb5b509b6499`.

## Recovery review

I replayed the preserved production diff against current main rather than
claiming the failed worker result. Review made two bounded corrections:

- I did not add `u8` to array-element C lowering. This change admits scalar
  spelling and scalar calls only.
- I select a double-valued helper for `cast_bool` on a float. Converting `0.5`
  to `int64_t` first would silently change true to false.

The self-hosted shadow emitter compiled successfully and emitted
`uint8_t nl_byte_identity(uint8_t ...)`, `nl_cast_int_from_int`,
`nl_cast_bool`, and `nl_cast_bool_from_float`. Its build log and generated C
have SHA-256 `36494247175a93ab34d017709c8654b1254d9002d18dfe3d2e7c611512ad2219`
and `39dc120021d427bed58d443863bd8f2ea3c46c9d4ab36c037d4ba654484ef781`.

## Qualification

`make -j8 bootstrap` completed Stage 1, Stage 2, both hello smokes, installed
compiler selection, and the no-C-seed smoke. The interactive bootstrap output
was not retained as a file, so I do not claim a sealed bootstrap log. The
resulting selected compiler SHA-256 values were:

```text
5e880ed5e406cd5052975141eae0dc235c3783faafd1b8e64d6083b845530197  bin/nanoc_c
a347afe521360a6de705c47ba082903500c953b6f55202d4cc9a10f337954037  bin/nanoc_stage1
e91f57c931ac4b6f8f4810cd590edce850c6ad97c58a259010d1c1cac97166cc  bin/nanoc_stage2
e91f57c931ac4b6f8f4810cd590edce850c6ad97c58a259010d1c1cac97166cc  bin/nanoc
```

The native Stage 1 and Stage 2 executable bytes differ. The bootstrap reports
that comparison without treating it as a correctness proof; canonical NanoISA
fixed-point evidence remains separate.

I then ran:

```text
NANOLANG_U8_C_COMPILERS=/opt/homebrew/opt/llvm/bin/clang \
NMS_NATIVE_CLANG_FLAGS='-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk' \
NANOLANG_U8_SANITIZER_LEAKS=1 \
python3 -m unittest -v tests.test_selfhost_native_u8
```

All four methods passed in 9.839 seconds:

- C-seed, Stage 1, and Stage 2 compiled and ran the exact scalar fixture;
- each compiler rejected the deliberately failing shadow without replacing
  prior output;
- each compiler rejected an out-of-range `u8` literal without replacing prior
  output;
- Homebrew Clang 23.1.1 compiled and ran the integer, `u8`, and fractional
  float conversion controls under ASan, UBSan, and LSan.

The full log is retained at
`/private/tmp/nanolang-u8-current-main-final/focused.log`, SHA-256
`76596fa6a13df2bd4f674bba0a82b9f954e8dade358b581ce236a4276acc0269`.
The eight tracked inputs and four selected compiler artifacts were identical
before and after the focused run. The retained evidence manifest has SHA-256
`eaffcf7ea113c7ba290a357775262cec4c79eddc57d8acfb3d12434a066ba901`.

This closes only the scalar self-hosted native-C spelling and conversion gap.
I do not infer array support, canonical NanoISA acceptance, full product
acceptance, or release readiness from this result.
