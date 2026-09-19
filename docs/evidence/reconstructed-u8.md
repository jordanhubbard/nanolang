# Exact `u8` scalar reconstruction evidence

I qualified this bounded change from production commit
`b70eefdb87c070cca753f9f78cd631ad35c06312`, based on canonical commit
`1e64676e16761936facc0c63da825257124a8184`. The checkout was clean after I
committed the implementation. I did not treat this work as a release gate or
as support for self-hosted native-C byte lowering.

## Qualified boundary

I reconstruct `PUSH_U8`, exact byte locals, direct byte parameters and
results, and explicit `CAST_INT` and `CAST_BOOL`. I preserve the values 0, 1,
127, 128, 254 and 255 through NanoVM, reconstructed C, reconstructed Nano,
three canonical source producers and translated native execution. I retain
prior output for a byte entry result, out-of-range and nonliteral contextual
source values, generic byte logic and generic byte comparison.

My self-hosted native-C compiler still needs the separately recorded
`task_633a3abf5a1040e6863a525ed5cc80b5`; Stage 1 and Stage 2 native-C output is
not part of this acceptance. Generic byte arithmetic, comparison, logic,
globals, imports, aggregates and ownership also remain outside this change.

## Final gates

| Gate | Result |
| --- | --- |
| `make -f Makefile.gnu -j8 bootstrap` | PASS; Stage 1, Stage 2, installed hello and C-seed independence passed. Native Stage 1 and Stage 2 binaries differed, as the gate reports; I make no fixed-point claim. |
| `python3 -m unittest -v tests.test_reconstructed_u8` | PASS, 5 methods in 7.797 seconds. |
| The same command with Homebrew LLVM 23 and `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` | PASS, 5 methods in 14.747 seconds with ASan, UBSan and LSan enabled for generated native programs. |
| The two adjacent truthiness/comparison refusal methods | PASS, 2 methods in 8.702 seconds. |
| `make -f Makefile.gnu -j8 test-scalar-reconstruction` | PASS, all 60 methods in 696.203 seconds. |
| `git diff --check` | PASS before the production commit. |

The final retained logs are:

| Log | SHA-256 |
| --- | --- |
| `/private/tmp/nanolang-u8-bootstrap-corrected5.log` | `fbdc6a4ccaeb0a949f0148840f642494f6cf5cda6e2c149d68bdc1cf222bf0f1` |
| `/private/tmp/nanolang-u8-focused-final.log` | `a1147da4b84221e5341dedbc78c3300c3acbf69e84ff4f12e16a3bc5884ec131` |
| `/private/tmp/nanolang-u8-focused-homebrew-lsan.log` | `0787600acadd71deeae37ebcbff2ee8ab89fb0eb6d68d04dde3adbe1f5e85a95` |
| `/private/tmp/nanolang-u8-adjacent-final.log` | `b80b33449b0f43633d040d422c6cf079ee9a6e52d1e94165541dfece55e6d0fe` |
| `/private/tmp/nanolang-u8-scalar-reconstruction.log` | `b4f7d1e5fadd36c6e8f0dc9a97859fb18549ddf431c44d122ae3c06c1b70879f` |

## Preserved terminals

I retained each first failure before its correction:

- `016aa15199f9b8e8ef26797548ec7c50e06c744e69b4d60153b573968cd0a16e`
  records the reserved `byte` shadow-local name.
- `9383e07e540f7f10678742298633d47a8e1dbe47e4d068a7c68f6e681666c62f`
  records the invalid assumption that self-hosted `NSType` had a `TYPE_U8`
  enum member.
- `65cb024e24ac4fc324fdb67fc682ff69264a15c2e14cc81161819ffd93a0eccf`
  records the first focused invocation whose tail was invalidated by a full
  host data volume. I count no short-write tail result as language evidence.
- `8ab2c9b2e920025f740824c7cc4ec9499b074360bd510792d2efc90d70317125`
  records missing `nvm2c`, the reserved negative-fixture name and the
  undeclared conversion call in a literal-only source control.
- `007b0e5bf5ffb1a2f6ef648ea8e131fc3a82ec14d33987285d4964fa2cdd0bd8`
  records the self-hosted `cast_int(u8)` whitelist refusal.
- `e22f0eca9d77ec505c8e6994ad8b5944635216c8e5b991f96855e3bbdfa9287b`
  records the self-hosted direct `u8` result refusal.
- `b26bd463644a62b3bd3cb4fa7672428b3dc495601473124f3f6b97685197f2a7`
  records the endpoint assertion against a correctly removed unreachable
  function. The corrected fixture makes that function reachable.

## Selected tool identities

The SDK-aware normal compiler was Apple Clang 21.0.0 at
`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`,
SHA-256 `1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`.
`/usr/bin/cc` is the Apple dispatcher, SHA-256
`b8763cf250e607a778bb4603cecb5b90338814d0a3dfcba0d57b1de242f610e9`.
The required leak-sanitized native gate used Homebrew Clang 23.1.1 at
`/opt/homebrew/opt/llvm/bin/clang`, SHA-256
`570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888`.

After the complete scalar gate, selected binaries had these SHA-256 values:

- `bin/nanoc_c`: `ccfdad221c17a8e3c9a0d6cad84832f26d9b03f9b1c5309d57b38b702eca9ede`
- `bin/nanoc_stage1`: `4e05901d238d5024442ac03ddfa525b5b405b53ec9e34f57db1db63d55e111a7`
- `bin/nanoc_stage2`: `06231967649f8408e4f02726bdc3f49cb010e017dbaf10fdd179ba066ae1faaa`
- `bin/nano_virt`: `ba4980b53e35d5581244f557c6b306ad50972fb511d1aaaabd6406e4433261a9`
- `bin/nano_vm`: `38d2f0b822439bb2fcb33cfe84e5ac88cb33d6369ad8dd04f1eaaa3190abdf3c`
- `bin/nvm2hl`: `c2e67d240f1c0028957d982ac2329b8ee51714d91f63c587a8a29113a8717ea4`
- `bin/nvm2c`: `42c396e47d5b7bc63dc90316b4b0c80599c0df3951272c306a71f5ce15a93b66`
- `bin/nanoisa`: `d13069d5aacdb99d445b7064618faf35f246dfcee461ecf76ceb9d10098208df`

The complete scalar target rebuilt some selected C tools before running its
methods. I therefore report these hashes as final selected identities, not as
a claim that every binary remained byte-identical across every earlier gate.

## Current-main integration

I restacked the branch onto canonical `179626d392349c26d7bd1eb165f29d7bd2c32c7b`.
The U8 production/test commit became
`40f6c55bd0a6ef215de2ac482dffd8854b912506`; its stable patch ID
`581a745317e45e077da71e7507c3fd3209b103c9` is identical to the qualified
`b70eefdb` production patch. The only files touched by both histories were
additive Makefile and roadmap changes; the rebase completed without a source
conflict.

I then rebuilt a fresh current-main C seed, Stage 1 and Stage 2. Bootstrap,
both hello smokes and installed C-seed independence passed. The exact five U8
methods plus the two adjacent generic-operation refusals passed 7/7 in 15.068
seconds. I did not repeat the unchanged 60-method gate solely for unrelated
canonical File/CODE/Forth integrations.

- `/private/tmp/nanolang-u8-integration-bootstrap.log`:
  `253c1e2a8761091a029f70656dbe3e8340473da9a45914d4bebbb547329a837e`
- `/private/tmp/nanolang-u8-integration-focused.log`:
  `cc130c992bcaaf8b87912d64c6dd03a2ace3f237888d90697a50d4120dd46aae`

## Direct parameter metadata

My final static review caught a defect that the executing controls did not:
the self-hosted emitters serialized an exact `u8` direct parameter as `void`.
At PR859 head `7ab0afc0`, the same fresh source produced these retained dumps:

- NanoVirt: `.parameters 0 u8`, dump SHA-256
  `01073092f2f8c609281f132bea8348abc1f6084f0fe3132eee07f134ebc41436`.
- Stage 1: `.parameters 1 void`, dump SHA-256
  `382939176dd30a30cab95645efa61bda56a4abbec243174646023553a49d845a`.
- Stage 2: the same incorrect dump SHA-256
  `382939176dd30a30cab95645efa61bda56a4abbec243174646023553a49d845a`.

The original probe source SHA-256 was
`529f2ff7f1a947d077e068e2f7603220a32cc318244cdbac60f25e364cdbc1c1`.
I recorded `task_affeba0b68cd8ad7e2c3756c672c934a` and the roadmap row before
changing production code. I now publish explicit parameter metadata when a
function declares a byte parameter and emit its exact `TYPE_CHECK 2` guard.
Other functions retain their prior metadata route.

The first corrected ad-hoc probe log
`cf4048a519fddd11d6056174cf700e171ccb7697fc05faddee087aec5b4f3515`
ended nonzero only because its `main` deliberately returned 255 while the
shell command expected zero. I preserved it and used a fresh source that
asserts 255 internally and returns zero; I did not reinterpret the terminal.

At production commit `8b11a889`, the corrected source SHA-256 is
`05bb76c8b38e636cc952e08fb872229f1c3cf58d9ae7351228d94aae0cfd1dbc`.
The qualified dumps retain `.parameters … u8` through every producer:

- NanoVirt: `8a1ae658d6116cb74044a8e63c298a759435f4c54d8972bf7f7388dc632946b9`.
- Stage 1 and Stage 2:
  `094225552754744f177f68ca96fccfd9e09715c668c90904dedd3bfb78f7c1b3`.

All three corrected modules verified, executed in NanoVM, translated through
`nvm2c`, compiled under Homebrew LLVM sanitizers and executed with leak
detection. The combined probe log SHA-256 is
`a4ecbb14a90f31db7d9dac9f2d52b7b40984e9065a90d569838a117f356f3b5d`.
Fresh bootstrap passed (`0f2437268f65f71bdd3d04ff4938b8cd5a8c5c279b0288058278b3d14d134d4e`),
the U8 plus adjacent refusal gate passed 7/7 in 16.941 seconds
(`f62e27336b4b8c1b0162f01c39ab133d360e0b1ef518f4766aa4ad850ef5f597`),
and the complete scalar reconstruction gate passed 60/60 in 570.949 seconds
(`0c2960d4ac2c0b99098bc80740a49c6127a14319069aa1c5feb0a512818d0d5e`).

## Latest canonical integration

I restacked the final correction onto canonical
`59ffceccf422047c91395e1cf5d08e62e5a1d64c`. The exact U8 production patch
keeps stable patch ID `581a745317e45e077da71e7507c3fd3209b103c9`, and the direct-parameter
correction keeps stable patch ID
`dda5e7192387b1a0aa6f0da564074cea8fc50853`. The rebase completed without a
conflict. Canonical changes to `src_nano/compiler/nanoisa_codegen.nano` and my
U8 changes are both present.

My first fresh integration checkout passed bootstrap, then a direct focused
invocation stopped before semantic execution because `bin/nanoisa`,
`bin/nano_vm` and `bin/nano_virt` were absent. The declared
`test-scalar-reconstruction` target already prepared the assembler, VM and
translators but omitted the NanoVirt producer used by its source controls. I
retain that setup result rather than calling it a U8 failure:

- bootstrap log SHA-256:
  `ce4ffeb9e929b82d15bf39a7127811cbf26fa460840958d0e39407838c282124`;
- missing-tool focused log SHA-256:
  `58208b8b31129d959ceebd8fd04e567fe1e96ab8f23c40e6e61b60e0bad889ee`.

I recorded `task_b9eb615aeba4676dbcf8d753e35960ee` before changing the
Make target. At correction commit `42619666`, `test-scalar-reconstruction`
prepares `nano_virt` alongside its existing prerequisites. In a second fresh
detached checkout the declared target passed all 60 methods in 592.770 seconds
after fresh bootstrap and tool preparation. A separate unchanged U8 plus
adjacent-refusal run then passed 7/7 in 13.025 seconds. The checkout remained
source-clean.

- complete target log SHA-256:
  `28866a5b62f562684a1d6e57ac7e386f48d55bc430f9efcce898eeb5c1b9b990`;
- focused log SHA-256:
  `d98c37cd8775be3a9dfcf9bb07346cf2eaf2e8c7436cede7bdaeeab8a787d5ac`.

The final selected binaries were `nanoc_c`
`3dcf35e11683ceee03229cee48f3736c0260b6c3591a460f9a82af5e6be914a1`,
Stage 1 `cee54d69e3490b5553a4143345d1c0c7128394f2b7b06936b7c2f8b9b972e362`,
Stage 2 `9625e6c7e410ddd611656ddfd87669a379b8907278912b75ee281688d010da95`,
NanoVirt `19b0d028cf0321169fe641d0bfa04bc5cabeb2aaa683058265702a43df2d23a2`,
NanoVM `fa47bae0787bb0bce80d85045a57c98f0ad089480b1d371859b77594543e57f3`,
`nvm2hl` `c2e67d240f1c0028957d982ac2329b8ee51714d91f63c587a8a29113a8717ea4`,
`nvm2c` `9756ac1367e2e81da83fab1ba088876ac4fb27a002c9beead919540836aca462`,
and the NanoISA tool
`ad0e54d01ca73bd6cd64b263d74675795f98ea3befedfd6edaf7e4aad487c1ad`.
I make no release claim from this bounded integration gate.

## Current canonical restack

Before merge review I fetched canonical
`8711402bef0dc290df5914cf0c08333563a7471b` and rebased this branch without a
conflict. The exact U8 production patch keeps stable patch ID
`581a745317e45e077da71e7507c3fd3209b103c9`, the parameter correction keeps
`dda5e7192387b1a0aa6f0da564074cea8fc50853`, and the scalar-gate prerequisite
keeps `40ad16887eb22136faf32d1012fe1a474ce276d7`. Canonical changes overlapped
only additive Makefile and roadmap history; no U8 production file changed
after the qualified `59ffcecc` integration base. I did not start another large
build while the Darwin data volume was under an explicit storage hold. The
existing fresh qualification remains pinned to its recorded source and tools;
this rebase establishes patch identity and mergeability, not a relabeled test
run.
