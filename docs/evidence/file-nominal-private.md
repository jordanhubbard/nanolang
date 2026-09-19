# I qualify private File nominal descriptions

I freeze source, fixtures and selected recipes at
`856620f83f44c9e4426b0fa77491705b61b9c602`. The five production files remain
exactly reviewed `8215045a9`; [my contract](../NANOISA_FILE_NOMINAL_TRANSPORT.md)
limits this checkpoint to a separate raw v2 codec, private descriptive query and
immutable catalog accessor. Task21469 remains open for later combined retention;
full File72556/6931 and handle/source/service parents remain open.

| First frozen gate | Status | Outer seconds |
| --- | --- | ---: |
| Linux normal included-query and linked fixtures | PASS | 4.836 |
| Linux GCC strict sanitizer pair | PASS | 1.651 |
| Linux Clang strict sanitizer pair | PASS | 1.509 |
| Linux old raw codec and File-plan targets | PASS | 1.999 |
| Darwin actual Apple Clang normal pair | PASS | 3.688 |
| Darwin Homebrew Clang strict sanitizer pair | PASS | 3.511 |
| Darwin old raw codec and File-plan targets | PASS | 2.283 |

Each included-query fixture passes19,746 checks; each separately linked fixture
passes19,735. Raw controls use independently written golden120 bytes, every
short length, unknown version/catalog/count/ordinal/reserved fields, high-bit
indices, duplicate/NO_INDEX mappings, encode/decode overlap, source overwrite and
failure output sentinels. Query controls use both direct and permuted catalog /
global / source-kind order, exact immutable names, flags and selected child
identities, then mutate every encoded layout/ownership byte and every service
payload byte. They check local/parameter/result counts and tags, borrow modes,
source counts, unsupported path version, wrong import kinds/signatures, extra
links/callbacks, name mutations, the layout limit and output preservation.

The included production query intercepts only its malloc/free, exercises its
sole allocation failure and recovery, and checks zero live query allocations.
The linked fixture compiles production query separately without interception.
Both destroy source modules before checking retained map getters. Bounds failures
leave getter outputs unchanged. The old shared authority validator/query,
general verifier, native emitter, v1 service validator and current bridge still
refuse the new metadata. Removing service bytes still leaves COMPLETE UNION
facts refused by shared ownership validation. No service instruction or host
File operation executes; fixture assembly uses only an ordinary scalar module.

The sanitizer recipes compile the new codec/query and immutable catalog provider
with strict C11 warnings and ASan/UBSan, leak detection enabled. Fresh supporting
NanoISA objects retain the normal project build flags; I do not claim those
older objects are fully sanitizer-instrumented by this gate. Linux Clang uses
its explicit GCC13 support path. Darwin selects actual Xcode Clang for normal
builds, Homebrew Clang for sanitizers and explicit xcrun SDKROOT. These normal
selected targets are wired into test-units, but I do not claim the full suite.

I seal57 reports in [report-sha256.json](file-nominal-private/report-sha256.json),
with2,246 tracked source/build/test inputs and six Linux/eight Darwin tool hashes
equal before/after and current in each frozen tree. The SDK identity is recorded
separately. Every phase archives supporting objects/dependency files and normal
fixture executables; each sanitizer phase also archives its exact linked fixture
executables and build/run commands/logs. The560 indexed artifact entries include
intentional repeated supporting objects across phases, representing182 distinct
content hashes. I verify every copied archive against its recorded hash.

Linux source is `/home/jkh/Src/nanolang-file-nominal-qualified`, reports
`/tmp/nanolang-file-nominal-856620-linux`. Darwin source is
`/tmp/nanolang-file-nominal-856620f83` on `CXWWHGGJX0.local`, reports
`/tmp/nanolang-file-nominal-856620-darwin`. The content-addressed local archive is
`/tmp/nanolang-file-nominal-artifact-store`. All first terminals pass; there is
no retry or assertion weakening in this checkpoint.

NvmModule has no serialized feature-bit envelope. These gates do not establish
v2 feature/section agreement, module version dispatch, container roundtrip,
source publication or VM/native host-granted execution. Converter, selector and
production provider lists remain unchanged. That combined transport checkpoint
requires independent review and its own frozen module acceptance.
