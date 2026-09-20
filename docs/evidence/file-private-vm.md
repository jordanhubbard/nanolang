# I qualify actual private File VM dispatch

I qualify productionbdc8d29f3 and corrected fixturecfb2a9903 under
task_e105bb912f9e44cd84e16ce78358b508, following
[my reviewed contract](../NANOISA_FILE_PRIVATE_VM.md). I execute copied checked
CODE through the macro-gated private adapter. My public VM/FFI/native/wrapper
routes still refuse File service modules. This is not native lowering, paired
source/shadow acceptance, installed execution or complete runtime82ff/5.1.

| Gate | Linux | Darwin puck |
| --- | ---: | ---: |
| Fresh common provider setup | PASS20.410s at ec54 | PASS9.293s at cfb |
| Ordinary instrumented and linked private VM | PASS6.872s | PASS5.688s |
| GCC ASan/UBSan/LSan | PASS23.280s | Not selected |
| Clang ASan/UBSan/LSan | PASS22.770s | PASS13.775s |
| Preserved frame/carrier regression | PASS21.814s | PASS12.399s |
| File opcode/public refusal | PASS3.073s | PASS3.549s |
| Ordinary wrapper publication | PASS3.180s | PASS6.666s |

Both corrected outer drivers exit0. The corrected Linux fixture reuses321 exact
hash-matched setup outputs from ec54; fixture sources alone changed. Every phase
builds and retains fresh fixture/provider objects and binaries. Puck uses a fresh
cfb source tree and explicit Apple/Homebrew compilers, SDK/libffi headers/link
stub and OpenSSL selection. No Darwin ec54 fixture ran.

Each new private VM run reports182,075 instrumented and17,163 linked checks.
The complete old carrier/frame controls run first and are separately labeled;
the new outer assertion increments their printed carrier count to84,506.
Instrumented full-chain allocation tests retain478 refusals and zero recovered
complete invocations. Both ordinary and sanitizer phases retain exact output
sentinels, host descriptor closure, attempted loader/open/fork counters and final
tracked allocation zero. Generation fault wrappers call the actual primitives;
modeled close failure really closes first, then injects an error report. Modeled
read/write faults follow actual one-byte progress; they do not simulate a real
device failure. Copied-fact unknown-mask tests are unit controls, not mutation of
an immutable externally supplied plan.

I exercise actual numeric/control operations and Result arms, temporary File
I/O, nested/overlapping calls, partial owner roots and scratch-publication failure,
formal borrow forwarding, initializer-before-entry and suppression after handled
close Error. Public refusal and ordinary wrapper gates remain separate. Linked
mode's zero interposed host counters do not imply no I/O; concrete attempted-I/O
and FD observations belong to the instrumented provider.

I explicitly clear inherited LSAN_OPTIONS, keep detect_leaks=1 and strict errors,
and retain the environment/commands. Sanitizer coverage is the selected rebuilt
thirteen allocating hosted providers, three NSI cores, carrier, private adapter,
vm_ffi sentinel provider and fixture translation units. Reused common/compiler/VM
objects are retained and identified but are not thereby whole-program sanitized.
Linux Clang's GCC13 selection is native-only. No assertion or sanitizer was relaxed.

## I preserve the first terminal

Frozen ec54 Linux setup passes20.410s. Its first normal run fails status1/3.538s
at the scalar-output conjunction after old carrier84506/frame42381 pass. The
actual failing value/case was not printed. Static review finds my new fixture
expected7 from unchanged `owner_module`, whose bytecode and old qualified test
return37. I record this before cfb's exact7-to37 correction and failure-only
expected/actual diagnostic. Root independently reviews it. No production change
or inferred product defect follows, and the failed binary is never replayed.
The complete corrected runs above pass the retained full output assertion.

## I seal source and artifact identity

[My manifest](file-private-vm/report-sha256.json) seals149 reports and14 equal
source/tool before-after pairs. Its5,608 artifact references resolve to949 unique
content-addressed files (684,570,462 bytes) under
`/tmp/nanolang-file-private-vm-artifacts`. Every artifact and downloaded puck
archive is hash-checked. Both corrected hosts have exactly equal9,090-file source
maps; all current sources and9 Linux/14 Darwin selected tool labels match the
retained maps. Some compiler labels resolve to the same actual binary.

Measured Linux trees are `nanolang-file-private-vm-qualified` (ec54 first terminal)
and `nanolang-file-private-vm-corrected` (cfb). Puck uses
`/tmp/nanolang-file-private-vm-cfb`. This evidence tree is not substituted for any
measured source tree. Later canonical/common-provider integration will be
reported separately without relabeling cfb sanitizer acceptance.
