# I separate OpenCL identity acceptance from an unresolved leak report

I qualify the repaired registry and three source readers in an isolated tree;
I do not claim all gates passed. My [seal](opencl-buffer-identity.json) retains
235 reports and72 artifacts/source archives across five terminal gates. I retain
every first failure, compiler command and before/after source/tool map. All maps
agree within their gate; libraries were rehashed after the final gate.

My final production and fixtures are byte-identical between source pina64f8cf4d
and docs-only Clang gate pin8eb347026. Production is the reviewed stable-slot
repair350c43138, arity correction72abaa16e and reader correction96efbea56 with
bounded200-character reader diagnostics. The unrelated OpenCL build-log400
precision was restored before execution. I changed no private GPU/service ABI.

## My measured matrix

| Route | Result |
| --- | --- |
| Linux GCC13 strict ordinary host + actual OpenCL GPU | PASS |
| Linux GCC13 ASan/UBSan/LSan host-only | PASS |
| Linux GCC13 actual GPU under sanitizers | semantic checks pass, exit1 LSan75bytes/3allocations |
| Linux Clang18 strict ordinary host + actual OpenCL GPU | PASS |
| Linux Clang18 ASan/UBSan/LSan host-only | PASS |
| PATH Clang23git setup | compile-only refusal from GCC-install-selection warning |
| Darwin/Windows | not measured |

Each complete host configuration runs ten registry/model cases,27 source-reader
cases across three readers and both unchanged array-boundary assertion sets.
I compile C11/O1 with Wall/Wextra/Werror. Sanitizer hosts use address+undefined
sanitizers, frame pointers, detect_leaks=1 and halt-on-error. Existing POSIX
function-pointer conversion practice is not ISO-pedantic conformance; I neither
enable a pedantic-conformance claim nor disable warnings. Model objects are
ordinary host allocations and no substitute for GPU evidence.

Each ordinary real GPU run passes48 buffer allocations/releases,24 OpenCL kernel
launches and384 exact integer observations. I explicitly call the OpenCL loader
in the fixture, require CL_DEVICE_TYPE_GPU and select RT_OCL; the platform name
containing CUDA does not mean I used the separate CUDA execution path. Actual
identity is NVIDIA GB10, GPU type4, NVIDIA Corporation, driver595.84, OpenCL3.0
CUDA, platform NVIDIA CUDA. My loader is libOpenCL.so.1.0.0 and installed ICD is
libnvidia-opencl.so.595.84; exact paths/hashes and nvidia.icd are sealed. Each
fixture releases its cached kernels/programs, queue and context explicitly;
that fixture cleanup does not establish a new legacy shutdown API.

## I retain the failures rather than relabeling the aggregate

1. Frozen634595907 stops after four setup passes at strict GCC model compilation:
   existing discarded fread and path diagnostic warnings. No fixture runs.
2. Frozen8f001216d also stops after four setup passes: my first400-character
   path precision exceeds the unified CUDA256-byte diagnostic buffer. No fixture
   runs. Reviewed correction uses200; read/cleanup semantics stay unchanged.
3. Frozena64f8cf4d has95 passing commands before the96th command exits1. Ordinary
   actual GPU acceptance and all GCC host checks passed. The actual GPU
   sanitized fixture prints its semantic success, then LSan reports75bytes in
   three allocations through unknown module frames rooted at clGetPlatformIDs.
   I do not count this command as passed.
4. The separated PATH Clang attempt stops before source compilation with the
   Clang23git GCC installation selection warning under Werror. I retain its
   identity and explicitly select installed Clang18 for the next gate.
5. The separated Clang18 gate passes all94 commands in3.071 seconds of summed
   command time, including host sanitizer checks and ordinary actual GPU. This
   does not rerun or erase the real GPU sanitizer failure.

My retained75-byte report is task_315caf01b3764a3eb655d825c9203e24. The fixture
calls dlclose before exit-time LSan, and no pre-unload module mapping for its
unknown allocation PCs was retained. Static call ordering establishes the
platform-enumeration call site but cannot prove the owner or proper lifetime of
those allocations. I applied no suppression, leak-detector disabling, repeated
failed artifact run or speculative driver cleanup.

## My contract audit remains bounded

Stable exact identities, generation exhaustion, checked rollback/quarantine,
copy bounds, all-argument validation, exact queried arity and no Set/enqueue on
refusal have host controls. Source-reader controls require one close, balanced
source allocation/free and no driver publication for each injected refusal;
normal/empty source bytes remain exact. Real ordinary OpenCL GPU kernel/copy
and release behavior is measured independently on GCC and Clang.

I have not satisfied complete real GPU sanitizer acceptance or established the
allocation owner in315caf. I leave87ca open pending the parent's precise
acceptance/canonical review, and leave315caf, d03c and platform/public-service
obligations open. The reader prerequisitef13e has successful strict and sanitizer
host evidence but still requires canonical review/merge before reconciliation.
