# My non-executing File nominal transport acceptance

I retain the exact 120-byte version2 catalog mapping through both module bridges
and the version2 container. I preserve File/OpenResult RESOURCE flags, all eight
nominal identities and per-kind ordinals, ownership bytes, and required feature
bits 1/7/8/9. Existing version1 bytes remain supported as non-executing metadata.
My shared authority validators and executing/dropping consumers still refuse
service claims. I do not grant host File access or publish callable source.

My contract is [NANOISA_FILE_NOMINAL_TRANSPORT.md](../NANOISA_FILE_NOMINAL_TRANSPORT.md).
This completes the bounded transport acceptance of task21469 and its reader9664
and wrapperab823 prerequisites. Parent72556/6931 and generated service execution,
source bindings, File/Socket/GPU and full release acceptance remain open.

| Frozen source | Target | Observed result |
|---|---|---|
| f5e07e5b | Linux / Darwin | Fresh bootstrap PASS251.959s /355.073s; first module gate FAIL5.492s /9.294s at valid120-byte container reader framing. |
| acc254bf9 | Linux | Fresh tools5.747s; GCC14.262s, Clang12.761s and adjacency9.795s PASS. |
| acc254bf9 | Darwin | Fresh tools4.078s and focused15.597s PASS; adjacency FAIL15.441s at embedded quoted-path wrapper publication. |
| 775b02408 | Linux / Darwin | Fresh tools5.344s /3.493s; wrapper controls PASS3.848s /10.349s, including the new250-byte alias component. |
| 775b02408 | Linux / Darwin | Linux remaining adjacency PASS2.845s. Darwin verifier passes, then owned-array fixture link stops before execution because crypto search path is absent (outer7.435s). Recorded explicit LIBRARY_PATH correction passes remaining descriptor/origin controls1.565s. |
| 0d057b49d | Linux / Darwin | Canonical830 integration fresh tools5.800s /3.348s; combined nominal/service/authority/wrapper gates PASS16.366s /23.987s. |

The final combined gates run two nominal methods (633 allocation-prefix and433
linked checks), three existing service methods (725/655/212 checks), the private
owner ARRAY authority method (1,403 checks), and all eight wrapper publication
methods. Wrapper controls execute ordinary return42 programs; no service module
or File host operation executes. The nominal CLI controls preserve existing
output and require refusal from C/LLVM/Wasm/facts/recovery paths. Direct controls
also cover VM/invocation/callable/core, FFI/COP no-dispatch and output preservation.

I instrument six fixture-linked units with allocation hooks and ASan/UBSan:
`nvm_format`, `nvm_v2_convert`, `service_bindings_module`,
`service_file_nominal_plan`, `retained_layouts`, and `nvm_v2_layouts`. Other linked
objects retain their ordinary build flags; this is not a claim that the entire
compiler is sanitized. Leak detection remains enabled. Allocation sweeps cover
attachment, temporary version2 adapter, and both bridges with cleanup/recovery.
Bridge tests distinguish borrowed from deep-copied lifetimes and destroy inputs
before checking the retained copy. The separate private codec/query acceptance
remains [sealed at856620f83](file-nominal-private.md).

I preserve all first terminals and their actual artifacts without replay.
The reader correction admits only exact56/120 framing before full version-aware
validation. The wrapper correction dynamically retains all117 quoted object
paths and the link command; it does not shorten the path fixture or drop objects.
The final merge adds exactly canonical830's private query and five Make lines;
transport/wrapper source and fixtures are unchanged from their reviewed pins.
I claim no new bootstrap at acc254,775b or0d057: f5e07 bootstrap remains a distinct
source pin, and later gates freshly rebuild the affected C providers/tools.

My [report seal](file-nominal-retention/report-sha256.json) verifies182 reports.
[Per-phase artifacts](file-nominal-retention/artifact-store.json) map878 entries
to485 unique content-addressed files at
`/tmp/nanolang-file-nominal-retention-artifacts`. Each runner saves actual binaries
before later relinks, command/environment/log hashes, object inventories and
source/tool before/after maps. All nine attempts preserve those source/tool maps.
Original and corrected trees have2,252 mapped inputs per platform; final
integration has2,258. Current checks verify all mapped sources and final binaries
in each retained tree. Linux records nine actual tool files; Darwin also records
actual Apple Clang, SDK settings and, where selected, the resolved libcrypto.
I do not equate the Apple `/usr/bin/cc` dispatcher with the selected compiler.

Darwin retained trees use SSH host `jordanh@CXWWHGGJX0.local`:
`/tmp/nanolang-file-nominal-combined-f5e07`,
`/tmp/nanolang-file-nominal-corrected-acc254`,
`/tmp/nanolang-file-nominal-wrapper-775b`, and
`/tmp/nanolang-file-nominal-integrated-0d057`.
Exact local/remote report paths and counts are in the sealed current inventories.
