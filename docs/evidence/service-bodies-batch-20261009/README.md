# I check nominal service bodies in both source routes

I base this batch on `2679b2a74` and track it in #989 and #982. I keep the full 5.1 scope open.

My C loader and Nano driver now run independent body checks after acquiring the complete namespace and retaining nominal annotations. I retain expression and call facts for exact catalog/helper signatures, File borrow modes, Result payload arity and exhaustive distinct arms, scalar fields, operators, assertions and return paths. I check uncalled helpers and generated shadows. Invalid bodies and explicit local/fact/depth bounds refuse before publication. C allocation-prefix failures reclaim their partial results.

Status zero means nominal checking only. I still refuse executable service publication: path-sensitive ownership, cleanup, independent lowering, grants and the full paired release gates remain required. Ordinary constructors, captured callables and other unsupported source forms do not acquire execution authority from this checker. My component Nano driver is C-produced development evidence, not fresh Stage1/Stage2 release qualification.

## My native call repair

The actual C-produced Nano checker probe originally fails native translation with `nrec_t versus nmap_value at parameter 1 of function 6`. A record global reaches a callee also called with direct records. I retain the exact record parameter convention, constrain the tagged payload separately, and use the existing checked unbox operation at the call boundary. I do not equate an optional wrapper with record storage.

My raw VM and sanitized native tests cover both caller orders, direct/global and tail calls, nested strings and arrays retained after global replacement and collection, and rejection of absent, scalar, wrong-field-tag and missing-field arguments. The record-local and map-global neighbors also pass: 13 methods with LLVM sanitizers. `test-nvm2c` passes 2,438 checks; its dependent shape, callable, pop, local-clear and driver controls pass as well.

## My input-route repairs

The import-body matrix catches two distinct problems. My Nano parser stores a complete qualified call name with an empty module prefix; my checker had added an extra leading dot. My C parser retained only one qualifier and refused a re-exported call such as `api.files.write_byte`. I now preserve the complete identifier path for calls with and without arguments. Service bodies retain their namespace until the independent lowering path can consume them, instead of falling into ordinary function rebinding.

I retain the original native shape failure and import-body failures beside this report. Separate fixture setup failures used the misspelled `TAILCALL` opcode and Apple's sanitizer runtime, which does not support leak detection. The corrected fixture uses `TAIL_CALL`; LLVM supplies ASan/UBSan/LSan. Older neighboring harnesses invoke literal `cc`, so their qualified run puts an LLVM `cc` shim first on PATH as well as setting CC. I have not weakened assertions or disabled leak detection.

## My validation

My final five-method paired run passes in 145.326 seconds. I type-check the complete generated binding including all five shadow bodies, the lifecycle/helper example, 17 malformed-body cases, local bounds, unsupported-source states and prior-output preservation. Both producers' checker probes run in NanoVM and LLVM-sanitized native code; three actual drivers enforce publication refusal. Five import-body cases cover valid helpers, wrong arguments, invalid imported shadows, distinct-module File identities and re-exported calls with and without arguments. I instrument the C body checker and parser with ASan/UBSan/LSan; other common objects remain ordinary objects. Every C fixture also exercises all three allocation-failure prefixes and the first successful allocation prefix.

I build the component driver with `bin/nano_virt src_nano/nanoc_v06.nano --emit-nvm -o /private/tmp/nanolang-service-bodies-driver-qualified.nvm`. I run the paired gate with `NANO_SERVICE_BODY_DRIVER_MODULE=/private/tmp/nanolang-service-bodies-driver-qualified.nvm NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-service-bodies-sanitize`. `paired-body-qualified.log` retains the terminal. The earlier 18-case actual namespace/import graph regression also passes with the development driver.

My parser unit suite and all five C plus nine Python wrapper-generation tests pass after the complete call-path change. My native regression logs retain 2,438 passing checks and 13 passing record/map methods. The user-owned guide fixture remains unchanged at SHA256 `c739aeb158c5b3e94c15d8de1232e4e1415a3f20b2e80fcf96fba39f1dedb976`.
