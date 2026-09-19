# I qualify a private NSI file descriptor plan

I freeze source/tests at `22d275e4`; private production is exactly reviewed
`fb4fb672`, following contract821f/04ca and pre-code INT-domain correction7fa5.
[My contract](../NSI_FILE_OWNERSHIP_BOUNDARY.md) fixes the exact current-schema
NlNsi input plus immutable internal catalog. I publish only an in-memory plan.
This checkpoint contributes to `task_6931ec89b210421e9827fecdbb459dbb`; the full
verified binding/Result/service-call boundary and d03c/ed702 remain open.

| First frozen gate | Status | Outer seconds |
| --- | --- | ---: |
| Linux GCC strict sanitizer pair | PASS | 3.091 |
| Linux Clang23 strict sanitizer pair | PASS | 2.929 |
| Linux normal GCC target | PASS | 1.770 |
| Linux six adjacent NSI targets | PASS | 4.864 |
| Darwin Homebrew Clang23 strict sanitizer pair | PASS | 4.812 |
| Darwin normal actual Apple Clang target | PASS | 2.034 |
| Darwin six adjacent NSI targets | PASS | 5.121 |

Each sanitizer pair runs the actual current NSI parser on the exact v0 fixture.
The instrumented query passes1,315 checks, including deterministic sole-allocation
failure and unchanged output; the separately linked production query passes902
checks. Both mutate all document field categories: identifiers/names/types,
counts including SIZE_MAX, missing arrays, parameter enums and alternative valid
ownership modes, optional/streaming flags, unsupported type payload fields,
method reordering/duplicate identity and member/case payload identity. Rejected
queries perform no plan allocation. I restore each mutation before the next.
I free the parsed input before querying every output method/type and byte-domain
facts; null/out-of-range getters and allocation recovery are covered. These are
bounded descriptor assertions, not a proof of all possible input memory safety.
The C API still requires valid caller-owned arrays and strings, as its header says.

The instrumented fixture includes the production query source after substituting
only its malloc; the second compilation links the production C separately with
no interception. Strict C11 warnings, ASan/UBSan and leak detection remain enabled.
Normal `test-nsi-file-plan` runs both unchanged fixtures through project CC/CFLAGS/
LDFLAGS and is wired into test-units. The explicit sanitizer target preserves
separate compiler selection. I claim the selected normal target, not full units.

I seal32 reports in [report-sha256.json](nsi-file-plan/report-sha256.json), with
2,191 tracked source/build/test inputs equal before/after and current for both
hosts, plus six actual tool identities per host. Darwin inventories include both
Homebrew and actual Apple Clang before execution. Manifests retain every command,
environment, status, elapsed time and log hash. Fixture executable/build/run
identities are separately labeled post-run hashes, not before/after binary maps.
Linux frozen source is `/home/jkh/Src/nanolang-file-plan-qualified`; Darwin is
`/private/tmp/nanolang-file-plan-qualified-22d2`. No gate failure was observed.

The adjacent targets are test-nsi-cap/shm/fabric/runtime/gen and test-nsi. Their
fresh C-only trees have no NanoLang compiler binary, so the optional source-client
branch is not acceptance here. No file service is called by the new query tests;
there is no generated callable source, schema extension, import admission,
service-dispatch change, bootstrap or public owned-result claim. The unchanged
adjacent NSI tests retain their preexisting scope.
