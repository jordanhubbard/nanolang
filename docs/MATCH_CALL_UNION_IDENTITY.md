# Checked generic union call identity

I track this correction in `task_a1d38d93616f4c48a3b8692b30226a82`.
My baseline is `af8809b32850454d06d9c1881c3a27b43f4c9d9c`.

I retained arm64 CI runs 35498179116 (PR 906), 35498613035 (PR 907),
and 35498722005 (PR 908). Each completes bootstrap stages 1/2/3 and then
rejects `scripts/userguide_snippets_check.nano:201` with E001 when a match
scrutinee calls `extract_json_from_marker`, declared `Result<string, string>`.
The raw job logs, API records, exact failure excerpts and hashes remain in
`/tmp/nanolang-arm64-ci-906-908-audit`. I have not replayed failed binaries.
These are observed compiler refusals, not attributed infrastructure failures.

Both C checker match paths read a call's `return_struct_type_name` but omit
its declared `return_type_info`. My existing checked-expression metadata
reader already resolves direct function, checked callable and field return
metadata. I will use that reader for call scrutinees just as I do for fields,
retain the registered generic base for coverage, and derive the concrete
instance name and payload metadata from its exact type arguments. I borrow
this metadata; I do not introduce an owner or change its lifetime.

I retain unresolved-identity refusal, exact BOOL guards, domain checks,
coverage, purity and resource rules. I do not substitute a variant spelling
for a missing scrutinee identity or rewrite the original failing source.

Before execution I require independent source and fixture review. My focused
controls cover direct generic Result and Option calls in statement and value
matches, two different payload instantiations, actual call-once traces,
conditional guards, wrong payload use and missing-arm output preservation.
I retain the complete existing C-seed match-totality controls, including the
unresolved nested match refusal. Fresh acceptance also compiles the unchanged
user-guide snippet checker and runs its existing check target, then the
relevant shared match controls and fresh bootstrap. Platform and provider
attribution stay explicit; this correction does not close full generic-union,
affine, bootstrap-equivalence or release acceptance.

## Retained preparation failure

My first external runner selected `make build`, which completes the CI
component build but does not produce the `nanoc_stage1`/`nanoc_stage2` paths
required by the canonical guard corpus. Linux passed the build, unchanged
user-guide tool compile and all 12 totality/policy methods, then ran all six
guard methods: one passed and five errored on the absent Stage 1 executable.
I retain the complete first terminal and products under
`/tmp/nanolang-match-call-8538-linux`; I do not call its `bootstrap` phase
label evidence of the distinct `make bootstrap` target.

My correction is external preparation only: verify prior provider hashes,
run actual `make bootstrap` on the retained source pin, then execute the full
guard corpus and previously unreached user-guide/full test commands. I record
new provider products separately and preserve the original passed phases.
No production source, fixture assertion or compiler warning is changed.

## Separate public VM fixture priority

I track the discovered neighboring fixture issue in `task_c2807d311e5d46938d5193b394bb6840`.
Both corrected runs pass all six canonical guard methods and all 40 user-guide
snippets. Their first full `make test` then reports 271494 checks passed and
one failure in `test_stack_slice_underflow`: expected status 2, observed 5.
Linux ends after 21.829 seconds and puck after 18.633 seconds. The original
assertion does not identify the opcode; I do not invent a measured opcode.

Static source establishes a contradictory expectation: my generated
`FILE_DROP_STACK` row has zero immediate operands and consumes one stack
value, so the general fixture includes it. My service-pending guard rejects
this bare File module with `VM_ERR_TYPE_ERROR` before activation and handler
underflow checking. I preserve that public boundary. The fixture correction
will require the exact refusal, no created frame, unchanged caller stack and
values, and all four local/caller combinations for this opcode. Other opcodes
retain the original atomic underflow assertions. I will print opcode and
case coordinates before a status assertion fails.

I require source review before fresh corrected VM/full-test gates. I reuse
only retained providers with verified hashes; I do not repeat passing
bootstrap, totality or user-guide phases for a fixture-only change.

## Darwin sanitizer selection terminal

At corrected fixture `c1409c5b5`, both hosts pass 274570 VM checks plus
substring/callback/heap/stack neighbors. Darwin's full `make test` next stops
after 473.474 seconds: all eight reference-evaluator transport methods abort
with `AddressSanitizer: detect_leaks is not supported on this platform.`
That fixture reads `NANO_NATIVE_TEST_CC`, defaulting to `cc`, independently
of my ordinary Apple `CC` and canonical-guard sanitizer selector.

I retain the first terminal at
`/private/tmp/nanolang-match-call-vm-c140-puck`. My proposed external-only
correction selects the already inventoried Homebrew Clang through
`NANO_NATIVE_TEST_CC`, preserving `detect_leaks=1` and all fixture assertions.
I will retain its new command/provider maps separately. This is a demonstrated
tool-selection mismatch; I do not call it an unexplained infrastructure event
or claim the aborted transport controls passed.
