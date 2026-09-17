# My explicit owned transfer evidence

I implement `task_026799f971e34152b6c6898e3de700d2` after PR549. I allocate
previously vacant primary opcode bytes 0x0b–0x0e to `OWN_MOVE_LOCAL`,
`OWN_STORE_LOCAL`, `OWN_PACK` and `OWN_UNPACK_LOCAL`. I change no existing
opcode value and do not implement extended-plane decoding. My schema, enum,
generated metadata, encode/decode, assembler and canonical reconstruction agree.
V2 transport preserves the instructions and required ownership declarations;
legacy serialization refuses the lossy conversion.

I distinguish unique owned stack values from scalar values and record
observations. A move invalidates its local. A store requires an exact nominal
destination without an outstanding resource obligation. Pack consumes every
field in declaration order; nested record fields require owned values.
Unpack invalidates the whole local and creates every field obligation in order.
I reject duplicate/discard/projection of owned tokens, implicit ordinary stores,
wrong nominal identities, repeated moves, replacement of live resources,
branch disagreement and unconsumed obligations. Owned returns transfer one
exact result; other locals and stack values cannot disappear with it.

My `nvm_verify_affine_function` entry point checks structural declarations and
actual affine bytecode dataflow. Ordinary module verification consults this
analysis before preserving its runtime refusal. Valid constructor, move,
store, nested unpack, owned return and balanced-loop programs pass this
verifier entry point. Their successful analysis still does not authorize
execution. The VM explicitly refuses the new instructions even in checked
fallback without metadata, and native translation refuses them before output
publication. Caller alias substitution, reference creation/access/end-region,
VM/native transfer semantics and paired producers remain required.

At the pre-restack checkpoint based on `1d24de49`, I pass:

- 187 transfer/codec/roundtrip checks and 275 with allocation-failure injection.
- The same 275 under focused ASan/UBSan with leak detection. I instrument the
  affine analysis/state, ownership/place helpers, verifier, ISA and native
  translator; other linked objects are ordinary builds.
- 300/419 existing bytecode checks and 157/182 local-state checks.
- 272,624 VM checks in both switch and computed-goto dispatch builds.
- 2,735 NanoISA checks, 33 schema checks, 2,412 native checks and 1,092 shapes.
- The existing declaration artifact control/refusal gate and the new owned
  artifact VM/native refusal gate, preserving previous native output.
- Two real Forth host build/load methods, five wrapper links and seven
  publication methods, confirming all explicit source-list consumers link.

My first sanitizer run found an error in the new roundtrip fixture: I freed
its borrowed wire buffer before the bridge copied the decoded code. I now
retain that buffer through conversion and view teardown. The underlying
borrowed-container API is unchanged. I preserve the initial failure in
`/tmp/nanolang-owned-transfer-sanitizers.log` and the corrected passing run in
`/tmp/nanolang-owned-transfer-sanitizers-final.log`. The raw store-refusal
fixture initially lacked its required operand and correctly hit the existing
stack guard; supplying the operand now tests the explicit unimplemented
handler. Neither initial fixture result is reported as a passing gate.

Other local logs are `/tmp/nanolang-owned-transfer-final.log`,
`/tmp/nanolang-owned-transfer-core-final.log`,
`/tmp/nanolang-owned-transfer-computed.log`,
`/tmp/nanolang-owned-transfer-schema.log`,
`/tmp/nanolang-owned-transfer-aot.log` and
`/tmp/nanolang-owned-transfer-host.log`.

My full ownership and release parents remain open. Float-record transport is
still the independently tracked task93574. No source borrow producer or
runtime ownership admission is enabled by this work.
