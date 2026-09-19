# My required File-service transport qualification

I qualify retained execution-significant bytes and explicit refusal. I do not
invoke a File service, generate a callable binding, or admit opaque handles to
VM/native/LLVM/Wasm execution. My parent File/Socket/GPU and generated Result
obligations remain open.

My reviewed production at c34c7324f is unchanged through 038bf80b3. Fixture and
recipe corrections have separate pins. I retain every first terminal; I do not
re-execute old failed artifacts.

| Frozen source | Check | Observed result |
| --- | --- | --- |
| c34c7324f | fresh clean bootstrap | PASS, 254.286s |
| c34c7324f | first focused setup | STOP2, 0.148s; missing explicit Linux Clang GCC13 selection, before fixtures |
| ba7d78d68 | corrected setup, fixture compilation | STOP; missing test `-Isrc`, no fixture executable |
| e8b847d8f | strict fixture link | STOP; missing executable-owned `g_argc/g_argv`, no fixture execution |
| 47f9254ef | GCC focused two methods | PASS, 7.273s including setup; unittest 1.941s |
| 47f9254ef | Clang focused two methods | PASS, 3.462s including setup; unittest 2.038s |
| 47f9254ef | first adjacent suite | STOP2, 2.065s; container 28 pass/1 failure, old negative calls newly assigned bit9 unknown |
| 038bf80b3 | corrected Linux adjacent suite | PASS, 9.189s |
| 038bf80b3 | Darwin Homebrew Clang focused two methods | PASS, 9.378s including setup; unittest 5.269s |
| 038bf80b3 | Darwin adjacent suite | PASS, 27.585s |

I record corrections before applying them. The unknown-feature probe moves to
unassigned bit10; the unknown section remains 0x7f and import-kind negative uses
MAX+1. My exact required-service section/feature/import matrix is unchanged.
There is no new bootstrap claim for later fixture pins.

Each focused compiler runs 725 instrumented allocation/bridge checks and 655
linked checks. The linked fixture uses ordinary production objects. The fault
fixture recompiles only nvm_format, nvm_v2_convert and service_bindings_module
with allocation interception, retaining the same strict C11 warnings and
ASan/UBSan/LSan settings. It covers every observed allocation prefix, recovery,
attach/output atomicity and both borrowed and copied bridge lifetimes.

The fixtures check the exact five-method catalog, required feature agreement,
import permutations/cross references, malformed partial claims, and all listed
public consumer boundaries. Actual facts, native-C, LLVM, Wasm, recovered-C and
recovered-Nano CLI calls return refusal, leave stdout empty and preserve prior
output bytes. Direct public FFI/COP calls preserve loader state and do not start
a coprocess. Ordinary neighboring modules still verify, roundtrip and execute.
I do not infer an external process's loader state from its CLI status.

Adjacent checks cover raw codec (586), private File plan (1315 instrumented/902
linked), container (29), imports (38), module (43), bridge (365), end-to-end (20),
allocation/callback controls, five wrapper checks and seven wrapper Python
methods plus the shared profile method. All selected checks pass on both hosts.

Each frozen run retains 2211 source inputs, actual resolved host compiler/tool
hashes, commands, logs and generated-tool hashes. Before/after source and host
tool maps match. Linux current verification matches every recorded generated
binary. Darwin's adjacent Make invocation regenerates embedded IR and relinks
nvm2llvm (adjacent log lines202–204): its earlier focused hash is retained, but
that intermediate binary was not archived before replacement. I do not call it
current or byte-identical. All seven final Darwin binaries match their recorded
adjacent hashes; the focused fixture executables are separately archived.

My report files are sealed in
[report-sha256.json](service-module-retention/report-sha256.json). Artifact
inventories point to content-addressed copies under
`/tmp/nanolang-service-module-qualified-artifacts`. Darwin transfers are checked
against remote maps; original remote artifacts remain in their isolated tree.
The first unexplained failures from other tasks are outside this qualification.

Mixed runtime integration with canonical PR819 follows in a separate tree and
requires another source review and bounded combined gates. This report alone
does not claim that integration.
