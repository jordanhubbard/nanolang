# I preserve observable concrete scalar-union source parity

I completed the full Darwin qualification at production commit
`ba8d75ae16eefb7378da9ace0bfd09058009da23` on canonical base
`e59fc09b591db977e53e4fef549d7615ad34e114`. I then rebased the identical
production patch onto canonical `99f390264b7e860ffad659a3852c9b66cbd73399`;
the current production commit is
`bcd59ce70e2394a08778ce2cb36c428df6fd13ca` and the evidence head is
`72ab69970bc53a34f95a8c214d39e3fa16f37cba`. The old and current production
commits have the same stable patch ID,
`a022392a8ee4def2ad48e30e5b330825052eb9b6`. The bases include PR893 and
PR909; the later base also includes public cyclic File and assembler-capture
work. My branch changes no `ARRAY_FIELDS` implementation or array-authority
file. It emits only the existing mandatory-understanding `UNION_VARIANTS`
kind 1 extension; kind 2 remains owned and qualified separately.

The accepted source uses `Choice<int,string>` and `Choice<float,bool>` in the
same module. Statement matches inspect concrete payloads and return from the
enclosing function. Value matches observe integer, Boolean, float and string
payload behavior. The fixture exercises both data variants, the empty variant,
repeated guarded arms, exact call/result transport and lexical payload scope.
Its exact output is:

```text
right
semantic-source-parity-pass
```

## My Darwin result

I ran on Darwin 25.6.0 arm64 with Apple Clang 21.0.0 for the project build and
Homebrew LLVM 23.1.1 for generated-native ASan/UBSan/LSan execution.

- Fresh `make -j8 bootstrap` passed in 383.98 seconds. Stage 1, Stage 2,
  installed-compiler hello and operation without `bin/nanoc_c` passed. The
  native stage binaries differ; I do not present this as fixed-point evidence.
- `make -j8 test-affine-scalar-union-source` passed. Its two methods took
  40.695 seconds and the target took 158.75 seconds including prerequisite
  builds. C-seed `nanoisa_emit`, a freshly compiled self-hosted emitter,
  NanoVirt, Stage 1 and Stage 2 independently published the source. Every
  artifact verified, executed in NanoVM, translated through nvm2c, compiled as
  strict sanitized native C and produced the exact output above.
- A second direct frozen-tool run passed both methods in 40.739 seconds. The
  eight selected permanent tool hashes were byte-identical before and after.
  The 418-entry relevant source map and clean worktree status were also
  unchanged. The refusal half covers five cases across all five frontends:
  wrong payload, cross-instance assignment, mismatched match result, incomplete
  match and escaped payload binding. All 25 preserve prior output and report a
  checked type, identity, coverage or lexical-scope diagnostic.
- `make -j8 test-affine-scalar-union-runtime` passed 546 ordinary checks, 856
  allocation-path checks and the VM/native sanitizer method in 1.40 seconds.
- `make -j8 test-nanovirt` passed all 90 code-generation tests in 0.90 seconds.
- With `NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`, canonical
  `make -j8 test-source-borrow-emission` built both metadata helpers and passed
  all 56 source-borrow methods in 616.729 seconds, 732.38 seconds including
  prerequisites. Leak detection remained enabled.

After integrating canonical `99f390264`, I rebuilt `nanoisa_emit`, NanoVirt,
NanoVM, nvm2c and the assembler/dumper from the current tree in 92.30 seconds.
The unchanged two-method source-parity suite passed in 41.952 seconds. The
current affine runtime gate again passed 546 ordinary checks, 856
allocation-path checks and the sanitizer method in 2.05 seconds. The later
canonical delta changes File cyclic runtime/schema and assembler capture; I do
not relabel the earlier broad 56-method result as a complete current-main File
qualification.

## My retained terminals

I preserve the implementation terminals in `docs/ROADMAP.md`. The final
integration also preserves these qualification boundaries:

- `/private/tmp/nanolang-affine-union-source-parity-integrated-adjacent.log`
  reaches existing resource-payload-union and imported dependency-closure
  requirements outside this scalar, import-free source contract. I neither
  weaken those controls nor count them as accepted here.
- The first broad Darwin adjacency run selected Apple Clang for generated
  native execution and stopped because that sanitizer runtime does not support
  `detect_leaks=1`. A corrected run kept the original driver compiler
  environment and selected Homebrew LLVM only through `NANO_NATIVE_TEST_CC`.
- Two direct-unittest attempts then exposed the established missing
  `obj/test_local_bindings` and `obj/borrow_shadow_names` prerequisites. The
  canonical Make target built both before the passing 56-method run. I do not
  relabel either incomplete run as semantic evidence.

## My sealed logs

```text
80824dd038177a254ee59c5a2dbec1c1023043a9cd123240a8103affda8cb4d9  /private/tmp/nanolang-affine-union-source-parity-pr909-bootstrap.log
221b32d922781d334ef394f6c5844160bbba1d845c0a08abf874a519f6fa8778  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/focused.log
5058fe2efea92b89453eb9e7d888e4021f8185320ebb965db8d41ba59bfbc385  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/frozen-focused.log
76c82c6236fb30d09784e50909a9ded35c92bf9c247cff1dbe632e99be2fe6d5  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/runtime.log
35e8be8325579c90caf49a84241cb46a6c95f4e146dbd292a8fc3addaa231fb1  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/nanovirt.log
dda6c1cd78860ba7cd1fa9b58bb8663ed358261ab005cfdc49df742e29a020ee  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/source-borrow.log
729ccd7d3c2adddecb25bd9fe0a4236d697829d4be3f294ff8341988bad502e9  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/source-borrow-homebrew.log
144990525fae04b3475860ea920cfc548a594c57557018170db09b69c36c63a3  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/source-borrow-corrected.log
34c42808ead2d55b61f12cd1646abd1b424c71e2d8f343134e4d3e4661d1dd78  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/local-binding-prerequisite.log
4643eefe06bd7ec1e44257f3806b9230a95473694a27affa8719ed3e5782a737  /private/tmp/nanolang-affine-union-source-parity-ba8d-seal/source-borrow-final.log
d2f760f616952d63a68dc79d0a23f27f2f897d700ab4ee768f3e1ebfe7f35b51  /private/tmp/nanolang-affine-union-source-parity-current-tools.log
160d3dc2543305f64d1bb05bd8a914c12a1a0266a32cfdc57a2f5ebfceb18b95  /private/tmp/nanolang-affine-union-source-parity-current-focused.log
a322ebc6e1ee77c496737ddca5f72cf9cb233b34e1ffd059d29c44e2157ed85a  /private/tmp/nanolang-affine-union-source-parity-current-runtime.log
```

The frozen focused source-map SHA-256 is
`26dcad0c59882e66803d7f1b3008706801e51e7d8402d67eabce2e6a8a5cc3de`.
The equal before/after frozen-tool map SHA-256 is
`7af63e899b94228aff96474c1061f5840b41736a64ed157f3551a7a9d47f9140`.
The actual Apple Clang binary SHA-256 is
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`;
the Homebrew Clang binary SHA-256 is
`570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888`.

This is bounded Darwin scalar-union source acceptance. Independent Linux
qualification, mixed `UNION_VARIANTS` plus `ARRAY_FIELDS`, resource payloads,
imported dependency closure, full product acceptance, PR522 and release
publication remain open.
