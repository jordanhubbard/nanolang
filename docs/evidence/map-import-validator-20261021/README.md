# Ordinary heap import validation, without execution admission

I add a standalone private module-taking query. It prepares fresh copied boundary
facts, checks ordinary caller-owned globals and collector membership read-only,
and publishes only a copied report. No public selector/handler, VM execution,
loader, native or publisher admission is added. Managed live allocation extents,
balanced ownership and exclusive access remain API preconditions; this is not a
validator for forged pointers. Runtime identity records carry conservative
singleton structural obligations, not recovered static provenance.

## Actual qualification

`python3 -m unittest -v tests.test_record_map_import` passes **2 methods,15.426s**,
**13,164 checks per dispatch**, with unchanged O1 ASan/UBSan providers. The exact
`environment.json` selects four unchanged original STRING→INT modules (seed,
C-hosted Nano, retained Stage1/Stage2) plus four separately produced source-global
inputs from the earlier lifecycle baseline. The tests execute original ordinary
bytecode by removing only the private test annotation, then restore that annotation
for the read-only query. Public annotated execution/serialization/native refusal
is retained. The standalone Make target runs the same harness and is wired into
`test-units`; `make-plan.txt` is a dry run, not an aggregate Make execution claim.

- Each original runs twice with explicit result consumption. Unconsumed results
  refuse import. Source-global initializer replacement preserves an independently
  retained old record alias; the query never resets globals or invokes GC.
- Actual global capacity, not high-water count, is scanned. A real LOAD_GLOBAL
  assertion reads17 above count0; unexpected non-VOID reserved slots refuse.
- Real positive buffered PURPLE, unrelated zero-ref buffered BLACK, and live
  unbuffered PURPLE after failed suspect-buffer growth are accepted unchanged.
  Duplicate/missing membership and non-idle/thread states refuse without repair.
- Shared identity under the same singleton succeeds. Matching present shapes do
  not authorize sharing across distinct origins; ambiguous historical origins
  refuse. Ordinary record ordinal0/layout1 succeeds; relabeling def_idx refuses.
- All five scalar leaves at depths1–3, empty wrong descriptors and actual cycles
  are checked. Ordinary vm_destroy/GC reclaims cycles after refusal. A valid
  63-frame tree has64 origins;259 is the conservative scratch bound, not a claimed
  constructible positive path.
- Counted-NUL keys remain distinct. Cached hashes, terminators, value tags,
  tombstones, duplicate keys and unreachable probe positions are corrupted in
  readable storage, refused, restored and accepted again.
- All67 allocation positions run transient and persistent denial plus independent
  retry, including table growth with80 collector-only identities. Caller state,
  collector headers, prior output and live allocation balance remain unchanged.
  Independent query peak is81,729 bytes, excluding125,812 live caller bytes.
- Import state4200/identity row16/report40; old VmState803992/VmHeap6312 unchanged.
  Actual representative reservation67,193,773 bytes/16,864,804 logical work reaches
  exact caps with positive-only whitebox reservation; +1 refuses atomically.
  UINT64_MAX refuses before validator-table allocation after boundary preparation.
- Old private runtime/observation/heap:5 methods32.649s, runtime2714 and observation
 19208 per dispatch, heap10318 checks. Old observation charges and runtime reserve
  outputs remain exact. All six legacy allocation/work rows are byte-identical;
  Analysis remains39840 bytes. Eight strict GCC16/Clang cases pass, with final
  harness strict checks repeated after the added independent peak assertion.

## Retained failures and corrections

The unbuilt count-based globals draft was corrected during review before the first
build. The first test compile omitted the NvmModule header before its forward
declaration; moving the private header first corrected only the harness.

The expanded O1 sanitizer harness then overflowed its stack. Retained disassembly
shows main reserving0x671160 bytes before calling other owner-sized helpers.
Test-only noinline separates functions owning ~804KB VmState objects. Provider
optimization, sanitizer instrumentation and stack limits remain unchanged.

Additional fixture failures were the wrong vm_array_set signature (corrected to
explicit retain/store/release), a BOOL negative that had replaced BOOL with BOOL,
and an overlap test assuming capacity8 where actual allocation was1 (now explicitly
provisioned by vm_ensure_globals). These failures are retained independently and
are not reported as product defects. No assertions or ownership checks were removed.

Raw artifacts remain in the paths printed by each gate, including final dispatch
folders `nano-record-map-runtime-5nl4r_qo` and `nano-record-map-runtime-hiqr6t3r`
under the system temporary directory. This compact receipt normalizes trailing
blank lines only; original logs remain intact. Provider/source hashes and exact
commands are retained. No frontend build/bootstrap or public-route change occurred.
