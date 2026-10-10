# I connect public indirect File execution and product consumers

I continue #989/#982 from `17dba6ae5` on
`release/5.1-completion-20261007`. My public contract is in
[the indirect API description](../../NANOISA_FILE_INDIRECT_PUBLIC.md).

I add grant-gated VM and native-emission entrypoints, a public report header,
ABI checks, the installed header/archive closure, and explicit CLI selection.
I route File product execution/publication and selected shadows through that
checked plan. Old acyclic/cyclic selectors retain their admission rules. This
batch does not implement paired callable source lowering or complete 5.1.

## Execution and installed consumers

My initial public-only corpus passes both methods in 171.474 seconds. After I
add source integration and copy-lifetime regressions, the expanded corpus has
90 exact serialized modules: 78 generated products and 12 intended refusals.
Private VM, public VM, native O0 and native O2 match byte-for-byte on 1,164
instrumented traces and 1,158 linked-provider traces. The manifests, full traces
and hashes are retained here. Six indirect query methods pass. Both unchanged
cyclic dispatch methods pass in 35.775 seconds. GCC accepts the changed sources
and all 78 generated public products; five workflow tests pass.

The expanded two-method command ends with one test-fixture failure after both
full corpora and installed execution/CLI checks complete: my reference-map
mutation pattern expects a UINT64 comparison, but emission uses a direct
integer comparison. I retain that failure in `corpus-and-initial-installed.log`.
I correct only the pattern and rerun the installed controls against the same
artifacts, using `recheck-installed.py.txt`; `installed-corrected.log` records
success. I do not relabel the failed aggregate command as a passing command.

The corrected controls compile public headers as C99, C11, C++11 and C++17;
execute the actual 256-shared-formal module through the VM and native O0/O2;
check zero fuel, revoked grants, duplicate/conflicting CLI modes, invalid fuel,
legacy/default refusal and unchanged output on emission failures; and mutate
public ABI revision, report sizes and a copied reference-map entry. Native
symbols exclude the VM and emitter. My installation target installs the actual
archive and 39 headers. CLI binaries are copied from the corresponding Make
build into the staged prefix; this is not a fresh full compiler installation.

`qualification.json` retains process endpoints, return codes, deadlines and
confirmed process-group cleanup. Nonzero CLI endpoints are deliberate refusal
checks; the corrected runner checks their expected statuses. Allocation and
ASan/UBSan instrumentation cover the listed rebuilt providers and generated C.
Common linked objects, the installed archive and staged CLIs retain their Make
build flags. I do not claim whole-program sanitizer instrumentation.

## Source integration defect and correction

Selecting the indirect plan for real source products initially fails seven of
the eleven C-driver methods before shadow execution. The target checker rejects
`PUSH_VOID; STORE_LOCAL`, my existing copy-local lifetime terminator, at function
0 byte PC83 in a generated binding. The ownership checker and runtime already
accept that operation for nonowner, nonformal copy locals.

I teach target analysis the same termination rule and forget any callable target
set. Regression modules accept a fresh callable binding, refuse a stale read,
and refuse attempts to clear an affine owner or borrowed formal. Both runtime
layouts and native optimizations exercise those modules. All eleven actual
C-driver methods then pass in 71.051 seconds. I preserve the seven original
failures, corrected log and first harness import error rather than treating
those failures as infrastructure.

## Compiler and release boundary

I build a fresh Nano compiler module with `bin/nano_virt
src_nano/nanoc_v06.nano --emit-nvm`, the explicit 30-second compiler shadow-suite
budget and Homebrew Clang. I translate that module with `bin/nvm2c` and link its
native form against the existing AOT runtime. The compiler artifacts and six
host libraries are hashed in `compiler-artifacts.json`. All 337 recorded File
host source/dependency hashes match current contents, including the new public
providers and product/shadow selection. The compiler module and native binary
are C-produced qualification artifacts, not a new Stage1/Stage2 fixed point.

All eleven Nano-driver methods pass in 207.594 seconds through both VM and
native compiler forms. I retain that run separately, using the normal 10-second
File shadow deadline. Exact-revision Linux/Darwin CI, paired callable source
lowering, mixed profiles, remaining Socket/network/compiler/backend work and
full release publication remain open. The user's untracked guide fixture keeps
SHA256 `c739aeb158c5b3e94c15d8de1232e4e1415a3f20b2e80fcf96fba39f1dedb976`.
