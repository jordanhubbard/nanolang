# I qualify private File and OpenResult lifetimes

I implement only the first private checkpoint of
[my File execution contract](../NANOISA_FILE_EXECUTION_BOUNDARY.md).
Production is exactly reviewed `099bab27a4907d99021ee9ffcbe51c7a3c38a65d`.
I freeze fixtures and Make targets at `a8c2e05338181fcae933f026fa534e29af7d8655`;
Darwin's corrected checkout is `ac8458007657e9604163699a98fb3f58d186953e`, whose
only subsequent change records the first setup failure. I preserve the full
execution/source task72556, File6931 and broader handle/service parents open.
Only bounded private child `task_b5a35ba1daa528b9f67209875d9b42f9` is qualified here.

| Gate | Status | Outer seconds |
| --- | --- | ---: |
| Linux normal pair | PASS | 1.344 |
| Linux GCC strict sanitizer pair | PASS | 1.605 |
| Linux Clang strict sanitizer pair | PASS | 1.426 |
| Linux adjacent File/capability/plan | PASS | 2.905 |
| First Darwin normal compile, no fixture produced | FAIL, retained | 1.060 |
| Corrected Darwin actual Apple Clang normal pair | PASS | 10.074 |
| Corrected Darwin Homebrew Clang strict sanitizer pair | PASS | 3.946 |
| Corrected Darwin adjacent File/capability/plan | PASS | 3.391 |

My first Darwin compile cannot find errno.h when directly invoking Xcode Clang
without SDKROOT. I retain its unchanged source/tool inventories and empty artifact
map. The fresh corrected runner selects the SDK returned by xcrun and records
its path, version27.0, SDKSettings hash and xcrun hash. Neither production nor
assertions change. I retain strict C11 warnings and ASan/UBSan with leak detection.
Linux Clang uses the explicit GCC13 support path; Darwin uses Homebrew Clang for
sanitizers and the actual Xcode compiler for normal targets. Manifests record
exact commands and environment. I claim the selected normal target, not all units.

Each instrumented fixture passes1,989 checks, including270 real temporary-file
opens and270 real closes with zero live resources. Its qualified adapter hooks
call host stdio, except the named allocation/I/O/error injections. It includes
production C with those hooks; this is distinct from my second fixture, which
links all three production C files unchanged and passes1,289 ordinary checks.
Both run in normal and sanitizer modes. The instrumented observations include
an unrelated live sentinel, actual EBADF after owned close, counted allocator
recovery, stale/cross-context generations, exclusive borrow epochs, moves and
selected-arm extraction, scalar service Errors, byte-domain refusal, both stream
direction guards, failed seek, capacity and counter retirement, partial creation
cleanup, saved close/rollback errno, first and secondary cleanup failures, and
idempotent finalization. An OpenResult.Error is still an affine value until
extracted or dropped. Finish revokes remaining borrows under explicit serialized
API preconditions; I make no concurrent-entry protection claim.

I seal67 reports in [report-sha256.json](file-value-lifetimes/report-sha256.json).
Each of the three runs retains2,232 source/build/test hashes and equal before/after
maps. Successful Linux and corrected Darwin maps also match their current frozen
checkouts. Six actual Linux tools and eight Darwin tools are equal before/after;
current maps for the successful trees are equal too. The corrected SDK identity
is additional to those tool maps. I preserve52 artifact entries in a content-addressed
store, including both normal executables, every sanitizer executable/build/run log,
and the distinct adjacent executables captured after their terminal. Normal
executables copied again by later steps remain explicitly named duplicates, not
new compilations. My artifact index records source paths, byte sizes and hashes;
no qualified executable is replaced by a later phase's tool.

Frozen Linux source is `/home/jkh/Src/nanolang-file-values-qualified`, reports
`/tmp/nanolang-file-values-a8c2-linux`. Darwin source is
`/tmp/nanolang-file-values-ac8458007` on `CXWWHGGJX0.local`, reports
`/tmp/nanolang-file-values-ac845-darwin`. I retain the first Darwin tree/report
separately. The local archive is `/tmp/nanolang-file-value-lifetime-artifact-store`.
These private C gates establish no service opcode, public File/Result authority,
VM/native execution grant, source binding, shadow selection, public escape or
bootstrap acceptance. Those remain later reviewed checkpoints.
