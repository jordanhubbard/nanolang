# My generated backend integration

I qualify source `a1bb1cb85e5551f9e5bb10491f1608877ef6b7b5` against
canonical `87e0cef0e2473b5a9987615d15f8a38fe2a0e445`. My original
[generated C](record-array-generated.md) and [LLVM](record-array-llvm.md)
qualification retain their original source, fixture and sanitizer attribution.

My incoming source changes are the reviewed VM callable teardown, empty code
append guard/header, ordinary native host-argument state, and separately
qualified Forth arithmetic. I compare 67 selected runtime, emitter, query and
fixture blobs exactly; my private generated implementation is unchanged.

| Fresh gate | Linux | Darwin |
| --- | --- | --- |
| Provider build and exact discovery | Pass | Pass |
| Generated C, both compilers, O0/O2 linked and observed | Pass | Pass |
| Exact C emission parity and LLVM emission | Pass | Pass |
| LLVM native, both compilers, O0/O2 linked and observed | Pass | Pass |
| Wasm, Node and Wasmtime, O0/O2 linked and observed | Pass | Pass |
| Private VM, global flow, execution plan and origin neighbors | Pass | Pass |
| Two original package ABI/reproducibility methods | Pass | Pass in separate supported-compiler continuation |

Each generated-C configuration retains 1,590 clean child statuses, 146 complete
fault-coverage records and 18,976 independent recoveries. Each LLVM native
configuration retains 2,474 clean child statuses and the same complete fault
coverage. Each host's Wasm gate retains 3,739 clean child statuses, 292 complete
engine-specific coverage records and 37,952 recoveries. I check all 73 products,
both allocation-failure modes, contiguous worker ranges and all 40 measured ABI
signatures. My original 240-second child limits and explicit phase budgets remain
unchanged. Restricted link inputs establish my generated products' no-VM
boundary; `nm` checks supplement that boundary.

I preserve two distinct Darwin setup histories. The first external extraction
helper selected an SSH-PATH Python without `tarfile.extractall(filter=...)`;
no source file had been extracted and no product had run. I retain its observed
terminal summary; I did not capture that interpreter's historical hash. The
corrected qualification uses an explicitly inventoried installed Python.

The first package phase inherited ordinary Apple Clang, while its unchanged
lifecycle fixture always requires ASan/UBSan and leak detection. Its child
aborted with the retained unsupported leak-detector diagnostic. I recorded
`task_f54e02f7eaca487c980fe298b1132791` and precode `bff63cc15`, rehashed the
original source/providers, then ran only fresh configuration and the two
failed/unrun package methods with my already-qualified Homebrew wrapper and
explicit Xcode sysroot. Both pass; sanitizer options and empty `LSAN_OPTIONS`
are unchanged. I do not replay any of the 16 preceding passing phases.
Thus my histories retain 36 selected phase terminals: 35 passes and that one
failure, plus the separate pre-gate extraction summary.

My [exact report index](record-array-backend-integration/seal-sha256.json)
contains 472 report files. The [CAS index](record-array-backend-integration/artifact-store.json)
contains 32,923 objects / 1,186,143,976 bytes, with 123,171 references and 72 equal
source/tool pairs. Every executed inner product remains retained. My final
endpoint checks rehash selected source, tool and provider files; they do not
claim an immutable entire SDK or operating-system library closure.

My durable archives are independently decompressed and checked on both hosts:

- Linux: 15,828 members, 204,326,918 bytes,
  SHA256 `9e7c36148528340fb26a4782f17466ec89a739a0491358ab43fd112241558221`.
- Darwin: 18,579 members, 186,285,937 bytes,
  SHA256 `c624ebf60f35b008908f5f465dd490896b1c14ddf5ad444559c6a730cdd4187a`.

My local combined CAS is
`/home/jkh/nanolang-qualification/backend-integration-a1bb-combined-seal`.
My complete Linux tree also has a persistent local copy: all 89,674 entries
match bytes, modes and symlink targets. I remove no original qualification data.
My persistent Darwin tree is
`/Users/jkh/nanolang-qualification/nanolang-backend-integration-a1bb-20260921`.
Archive locations and verification records are retained under `closure/`.

My [independent original LLVM audits](llvm-root-integrity/README.md) distinguish
integrity, native coverage and Wasm coverage. My
[independent integration audits](backend-root-integrity/README.md)
check local integrity and selected native/Wasm semantic coverage; generated C,
startup, package and source correspondence remain separately scoped. Their
Git-publication correspondence is a subsequent check. My complete integration
coverage audit is retained separately for each host.

I do not claim installed publication of the complete generated consumer,
public selection, paired producer/bootstrap acceptance or full
union/nested/cyclic/indirect graphs. Those parent requirements remain open.
