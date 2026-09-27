# My self-hosted scalar-global checkpoint

Both owned source producers now emit explicit scalar-global initialization,
loads, checked mutable stores and ownership format 4 extension kind 3. Entry
initializes each slot in declaration order before production or selected shadows;
helpers share those slots. Locals and parameters retain lexical precedence.
The public verifier still refuses early reads/calls, wrong tags, aggregate stores
and incompatible managed-array/borrowed-helper profiles.

My fresh bootstrap completes Stage 1, Stage 2, configured component checks and
installed-compiler smoke checks. The unchanged exactly-once counter passes all
three compilers. The combined selected-pattern, selected-ownership, generic and
scalar-global matrix passes all 67 methods. My new source tests include eight
accepted programs and nine refusals through both native stages, preserving prior
output on refusals. C production executes mandatory shadows; the raw self-hosted
production emitter intentionally does not. I separately build its shadow-module
driver and run the eight accepted programs plus the failing effect assertion:
eight entries succeed and the deliberately false assertion traps. Native stages
refuse publication for that failing assertion. Production programs also pass VM
execution and generated native ASan/UBSan with leak detection.

The existing two owned-union source methods pass, retaining the original
13 accepted/nine refused corpus across both producers and match-arm name checks.
The compiler build executes imported shadows, including the new global lookup
and production/shadow-entry checks. The shadow policy checker passes.

I retain two fixture corrections. First, the new test incorrectly expected raw
`nanoisa_emit` production to execute shadows; I added explicit shadow-module and
native-driver checks. Second, direct shadow tests used a stale assembler that
refused format-4 declarations. My first rebuild named the binary rather than its
Make target and failed. I fixed the target dependency to `nanoisa_dump`, rebuilt
it, and reran the identical assertions successfully. The old/new binary hashes
and failed terminals remain here; I do not treat their failures as passing tests.

I also pass Make's selected compiler, C flags and link flags to the private owned
ARRAY descriptor fixture. Its unchanged 239 assertions now pass on Darwin with
no `LIBRARY_PATH` override. This qualifies fixture linking and descriptor checks;
it does not enable pending ARRAY execution profiles.

`provenance.json` pins source hashes and the parent commit. `logs.json` pins each
completed terminal and exit status. The complete source-borrow and scalar-union
source gates are still running at this checkpoint; their completion must be
recorded separately. Current-source fixed points, retained compatibility fixes,
complete platform/sanitizer gates and release documentation remain required.
PR #522 remains draft. Bootstrap smoke success is not release qualification.
