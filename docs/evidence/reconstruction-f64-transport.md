# My exact float reconstruction acceptance

I tested generator production `77653977`, based on main `cc0688f1`, under
`task_b6083f68f2cf4951a134da4ec82a1e17`. My [contract](../NANOISA_RECONSTRUCTION_F64_TRANSPORT.md)
preceded implementation. I admit exact PUSH_F64 constants and F64_FROM_BITS /
F64_TO_BITS transport, FLOAT helper signatures and locals, with the existing
immutable snapshots and structured local joins. I keep entry INT and preserve
all existing operator and structural restrictions.

I recovered every constant through its fixed 16-hex-digit `f64_bits` fact, using
integer decoding. Standalone C uses private memcpy helpers; Nano uses the merged
bit intrinsics. No floating equality, decimal formatting or host-float parsing
serves as an acceptance oracle. Exact integer bits determine success.

I passed the following ordinary gates:

- Seventeen binary64 classes in five small independent programs: signed zeros,
  subnormal/finite boundaries, infinities and signed quiet/signaling NaN payloads.
  I additionally checked the all-ones and maximum signed-integer NaN patterns.
- Original verified VM/native execution, reconstructed standalone C under
  GCC/Clang ASan/UBSan, reconstructed Nano through C-seed/Stage1/Stage2 legacy
  execution, and C-seed NanoVirt plus Stage1/Stage2 canonical module production
  followed by VM and generated-native execution. Generated native C is also
  sanitizer-instrumented. Legacy compiler-selected C toolchains retain their
  configured defaults.
- Float helper arguments/returns, snapshot-before-store, DUP/permutation use,
  both exact local-based branch arms and a bounded loop with pure bit conversion
  in its condition. A separate generated-source observer counts two helper calls,
  including one discarded return. I run that C/Nano observer through the same
  producer routes; it adds no global/counter opcode admission to reconstruction.
- Ninety-four FLOAT refusal controls spanning all 47 previously admitted operator
  mnemonics. Known float arithmetic, casts, truthiness and generic comparisons
  remain refused. Valid modules with a float entry, mixed local types or nonempty
  stack joins retain previous output. Missing/nonhex/wrong-width facts and
  unmapped source tags refuse before rendering.
- Existing integer/bool snapshots, both diamond branches, retained local names,
  exact binary64 facts/text and source-output preservation controls.

The first four-method GCC gate passed in 52.560 seconds; the expanded five-method
Clang gate passed in 54.376 seconds. Updated snapshot/discard observers and
boundary controls passed two GCC methods in 18.737 seconds, followed by the
integer-pattern endpoints and operator controls in 8.808 seconds. Three final
Clang methods passed in 27.120 seconds. Four selected existing scalar methods
passed in 33.518 seconds; seven facts/text/refusal integration methods passed.
These separate runs cover all seven current float-reconstruction methods without
claiming a single monolithic invocation or a full reconstruction suite.

My [manifest](reconstruction-f64-transport.json) distinguishes generator source
from reused qualified source compiler tools. I use the completed PR720 producer
checkout at `f87267c2`; its four compiler/producer binary hashes and one embedded
shared-library path stayed unchanged throughout the gates. I did not rebuild
those producers from the reconstruction branch or claim hermetic relocation.
My reconstruction branch builds its own facts reader, assembler, VM and native
translator. I preserve the actual compiler path and command in each retained
ordinary fixture log.

I preserve the historical PR720 blanket reconstruction refusal as evidence for
its original pin. Current refusal controls use still-unsupported F64_NEG instead
of newly admitted constants/bit copies. No historical PR679 endpoint corpus,
failed compiler or frozen product candidate was executed or modified. I make
no full-bootstrap, Darwin, full-product, float-arithmetic or full-reconstruction
claim from this bounded child; parent `4bd034f6029b7458201db74e2c3aeb32` stays open.
