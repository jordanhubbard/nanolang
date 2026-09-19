# I keep ownership truth separate from my public-C resource boundary

I record this preimplementation amendment under
`task_bd8ebe2ba75457af1686f095275229dd`, a child of installed-product task
`task_e8d860a16da0464891dd32e91c42bef1`. It follows the retained first Darwin
outcome at frozen source `254fa51e8dd49bb71b15c65406a355d172fcaf5d`.
I have not replayed that failure or run another fixture while writing this
contract.

This amendment supersedes only the exact36-source public-C expectation in
`AFFINE_OWNERSHIP_PROFILE_EXPECTATIONS.md`. I preserve every source byte,
case name and semantic `accepted` value. I add no resource C representation,
callable ABI or resource-union ABI.

## I audited the reached phase and diagnostic precedence

The C-seed driver completes `type_check` and dependency/root shadow checking
before entering `c_backend_emit`. A semantic negative therefore remains a
frontend ownership refusal and never becomes a public-C profile result.

For a semantic positive, `c_backend_emit` creates a sibling staging file and
calls `render_source`. Its namespace-planning pass calls `emit_program`, whose
first pass visits root items in source order. Every exact case begins with:

```nano
resource struct FileHandle { fd: int }
extern fn consume_handle(owned: FileHandle) -> void
```

`ctx_profile_node` reaches `cb_node_storage` for that first
`AST_STRUCT_DEF`. The supported-types policy rejects `is_resource` before
emitting a type definition. The first retained diagnostic is:

```text
I require a supported exact C value representation.
```

Planning stops at that root item. No function signature, body, indirect call,
later ordinary record or later union declaration is visited. On refusal the
path API removes its sibling staging file instead of renaming it over the
caller path. That implementation establishes the intended publication
boundary; my first failed harness did not observe the caller bytes before its
assertion terminated, so I do not use that run as preservation evidence.

This order agrees with `PUBLIC_C_SUPPORTED_TYPES_CONTRACT.md`: ordinary local
nominal records and bounded unions have exact context-specific C
representations, while unsupported root declarations receive checked refusal.
A source-language resource declaration is not an ordinary local record
declaration.

## I retain all14 semantic positives

Root independently compared the original and proposed36-case manifests: all
names, source bytes and14 positive expectations are identical, with sorted-key
JSON manifest SHA-256
`ab3b6dfea5c007358eb8fd4b4a2e97a71fb3c9a5373bc0a860c42bd10e6c2147`.
My frozen254 audit records each positive's complete generated-source hash:

| Case | Exact source SHA-256 | First public-C root result |
|---|---|---|
| `both_arms` | `5b42b7989956a7dc787ff6ed21316836da11c9281c9fa4201e6b410e75048747` | resource declaration refusal |
| `resolved_return` | `3abd916ef609ed8e25569e19e759bde890ce4f2ce23922eafc020559b69da0af` | resource declaration refusal |
| `indirect_result_owner` | `744bff273790a98e95255372cc833f032146f02b40bfd5ce620fa83ee22bcf7b` | resource declaration refusal |
| `loop_break_True` | `c332cae3f4f10811ca3b4cd552becc5141b06e6f2c6069dc686f368cf16d7311` | resource declaration refusal |
| `loop_continue_True` | `e832166e7ce29fb1e8fede5b04a4935484e3d0a89149b79fb34b161051657d86` | resource declaration refusal |
| `loop__True` | `7d9c01e6cfec6ad919b9a85621508d5923814b146aa0afc6ef01987aef913244` | resource declaration refusal |
| `move_consume` | `2e3c34b74c1fe69daa30327917afe3bf629b55ddac2806a0ed5a362b8eb3b0f1` | resource declaration refusal |
| `nested_return` | `95674ea51215920bcb888053ea6c36ac9962714b1019917499411d85c8dca5da` | resource declaration refusal |
| `observe_return` | `ee08e2aab8fcb164612e7b49e3a420b98d2fb03464211790069ecc629353e58e` | resource declaration refusal |
| `overwrite_True` | `72132888182eb734703f779c997518aa26c9dceb19a5d5bbc506adba3a2093cf` | resource declaration refusal |
| `return_parameter` | `a6b2e1e17432e95da23af725411fcad9f2ad4ceefa34cc0b6f425f317e8bf9b2` | resource declaration refusal |
| `ordinary_shadow` | `dc82d7e545ef5b0688bd022ea1faa449db71955847140081c5ae1dd443f2f942` | resource declaration refusal |
| `unconditional_left_move` | `18f9bbd5de85dfbe4bb7ebd3354f880add864e6664dc2ced713c9a82f1c9ecbc` | resource declaration refusal |
| `union_return` | `dc178337029b71d09558e1e876125b55a699577d7318dbad255df9d354e35f22` | resource declaration refusal |

The route authorities become:

| Authority | Exact expectation |
|---|---|
| Test-only actual C frontend | All14 remain accepted after full parsing, imports, type checking and dependency/root shadow checking. |
| C-seed public C | All14 refuse the reached resource declaration with the exact supported-representation diagnostic and preserve the known prior output. This is a backend-profile result, not an ownership rejection. |
| Stage1 and Stage2 explicit C | All14 retain their original positive expectations. A failure is a self-hosted transpiler defect and cannot be hidden by the C-seed profile result. |
| Canonical checked selection | All14 retain the existing three-produced Nano checker, Stage1/Stage2 `--emit-nvm`, verifier, VM and strict native expectations. |

The22 semantic negatives retain their original ownership expectations. The
actual C frontend and all three compiler drivers must reject them during
checking, before any backend publication, and preserve prior output where the
driver accepts an output path.

## I keep deeper public-C contracts independent

`indirect_result_owner` contains a callable parameter and indirect call.
`union_return` contains a resource-bearing union result. Neither construct is
reached by the exact-source public-C route because the common resource
declaration refuses first. I therefore remove their diagnostics from this
affine corpus's expected public-C table.

I do not weaken `PUBLIC_C_CALLABLE_PROFILE_CONTRACT.md` or
`PUBLIC_C_UNION_RESULTS_CONTRACT.md`. Their fixtures omit this common resource
prefix and continue to qualify their own reached callable and bounded union
profiles, publication preservation and same-process recovery. Historical
logs that reached the deeper diagnostics remain historical evidence; they are
not relabeled as current exact-source outcomes. This amendment neither claims
that the deeper diagnostics disappeared nor requires them to become reachable
through a resource declaration.

## I observe output before an assertion can stop the harness

The first route failure occurred after the subprocess returned but before the
harness read its output path. `TemporaryDirectory` then removed the case. I
correct the harness sequencing before another run:

1. Each route receives one unique case/authority directory and a known prior
   output byte sequence.
2. Immediately after `subprocess.run`, before any assertion, I snapshot return
   status, stdout, stderr, source bytes, output existence and output bytes.
3. I hash the source, prior output and observed output independently. A missing
   output is a distinct observation, not an empty file.
4. Any nonzero publication result checks the observed output against the prior
   bytes before checking whether the refusal was expected.
5. Route assertions run inside `try`; `finally` writes the complete observation
   and copies the source/output into the qualification evidence root. An
   unexpected assertion therefore cannot delete the evidence it describes.
6. A qualification command supplies an explicit evidence root outside the
   temporary source tree. Ordinary local runs retain a newly created failure
   directory and print its exact path. Successful local runs may clean their
   temporary case directories after observations have been consumed.

The observation record includes the case name, authority, exact argv, status,
elapsed time, source/prior/output hashes, stdout/stderr hashes and the selected
compiler hash. I never execute a refused output artifact.

## My planned test-only correction

After independent review I will:

- replace the two-name affine public-C exception map with the one exact
  resource-declaration expectation shared by every semantic positive;
- keep the14/22 semantic counts and exact source manifest assertions;
- keep the C frontend runner unchanged;
- retain Stage1/Stage2 explicit-C and canonical selection expectations;
- add the observation/finally boundary above without changing production;
- run the36 C-frontend and route cases from a new frozen checkout;
- run the existing callable and union-result public-C tests separately;
- run the existing checked-owner selection and affine-selfhost gates; and
- defer fresh installed-product acceptance to its separately coordinated pin.

I will stop at the first new terminal, preserve it and record any additional
dependency before correction. No historical failed artifact is replayed.

## What this amendment does not close

This amendment does not add resource declarations to the public-C profile. It
does not establish a callable or resource-union C ABI, change ownership
semantics, weaken shadow checking or close remaining Samples, owner-array,
installed-product, full ownership or release-publication work.
