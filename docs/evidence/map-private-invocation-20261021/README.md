# Private map invocation qualification

This implements reviewed contract c13bc34d6 above heap component f8c2f127f. The closed private owner now invokes the actual VM through a call-local map context. Public verification, serialization, native generation and publishers still refuse selected map authority. No public observation API is added; tests use private stats and explicitly whitebox access.

The existing four-argument scoped core remains a NULL-map wrapper. Only the private supervisor and its whitebox controls use the five-argument context. Old VmState/VmHeap declarations and old array tables remain unchanged. Context identity, copied recipe/frame correspondence, operand tags and duplicate retain headroom are checked before mutation. Initializer and entry reset root identity/base separately; ASSERT resumes the same context and root tail replacement updates its identity. Failure unwinds stack/frames, invalidates the automatic view, and preserves the prior result edge. Shared global mutations are not transactional rollback.

## Actual gates

- First switch gate: 1 method, **7.827s**, 1346 checks. Four retained original STRING→INT modules (seed, prior C-hosted Nano, installed Stage1 and Stage2) execute result0 with their actual assertions after original module and source-plan destruction. Final tracked allocations are zero.
- Extended switch: 1612 checks, **7.500s**; initializer heap-result residue and false assertion, absent GET distinguishing VOID from INT0, root tail with17 duplicate record arguments and ASSERT resume, all execution allocation positions in transient/persistent failure modes.
- Context/alias switch: 1702 checks, **7.573s**. Prior result retains exact record identity while a failed second invocation has mutated its shared map field to2. Map/record retain headroom refuses before stack mutation.
- Switch/goto matrix: 1749 checks each, **14.905s**; invalid context epoch/root/active/verified state, busy/overflow invocation, corrupted recipe/result-mask/map operation, preserved result and valid retry.
- Corrected full matrix: **2715 checks each, 15.178s**, including40 keys, distinct counted `a\0x`/`a\0y`, exact HAS/GET/LEN, table growth, all execution allocation failures and retry, duplicate-tail UINT32_MAX-1 headroom with unchanged frame/stack/root on refusal. Both actual dispatch macros are verified in the final run.
- Final runtime + old array linked/instrumented switch/goto + preparation-only neighbors: **6 methods PASS61.735s**. The preparation-only build still has no invocation API. Listed rebuilt providers and whitebox harness use ASan/UBSan; retained common providers retain their recorded provenance.
- Six old budget rows are byte-identical, including Analysis39840. Ten strict Clang/GCC16 syntax cases pass (ordinary, array-only, preparation-only, runtime and harness).

Actual runtime owner818512 and automatic view48 are positively reserved; Stage1/2 preparation is9243432 bytes/9266634 logical work, with76473835/26142304 overlap. Seed and prior C-hosted rows differ slightly and are retained individually. No guest-work budget or runtime observation API is introduced.

`test-record-map-runtime` runs both dispatch modes and is required by test-units. Synthetic runtime controls always run; RECORD_MAP_MODULES optionally adds the separately retained original modules. The qualifying local runs supplied all four exact inputs. The driver inherits compiler/instrumentation and common object selection; the retained environment records exact inputs. No frontend/bootstrap build was performed.

## Failed attempts and limits

The first growth fixture had counted length3 but left a nonzero byte after its `a\0x` payload. The independent proof correctly refused "I require terminated counted strings." Correcting the fixture's trailing terminator preserved the embedded NUL and produced the passing matrix; no product change was made.

An additional ordinary-admission test with explicit O1 ASan/UBSan harness instrumentation failed before guest execution with stack overflow in main375. Actual disassembly reserves0x86a000+0x640 (8824384 bytes), beyond Darwin's8MiB stack. The exact HEAD vm.c baseline reproduces it. Its first temporary harness-copy compile lacked the existing fixture quote path; that failed attempt and corrected baseline are retained. Harness-only no-inline execution with identical sanitized providers passes, as does a separately labeled uninstrumented-harness/link-runtime control. Neither is claimed as an O1 sanitizer pass. The normal target does not forward ORDINARY_ADMISSION_CFLAGS and has no discovered test-units dependency; no default harness policy was changed.

This is private VM qualification only. Native key boundaries, public selection and source publication remain separate work. Readable logs normalize trailing whitespace; raw gzip copies retain differing originals. No binaries are bundled.
