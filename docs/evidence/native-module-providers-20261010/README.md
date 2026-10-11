# Native module providers

I continue the 606 duplicate-symbol compiler link failure retained in my
[interpreter-dispatch evidence](../interpreter-dispatch-20261010/README.md),
from 1fbe41036. The ordinary objects for file_product and file_source_inputs
bundled support already provided by nanoisa, compiler_support and the host
UTF-8 runtime. Manifest dependencies previously propagated headers only.

I add module_build_native for native C-seed composition. Its builder owns one
ordinary build result per physical module for the compilation. Each returned
result independently owns its object path and combined dependency link flags.
I visit each provider once per root closure, including cycles/diamonds, and
bound depth/module count. Missing manifests refuse native publication.
module_build retains its existing shared-library generation behavior.

I move overlapping support into shared_c_sources, preserving hidden private
copies in the dynamically loaded libraries. Native file_product declares
nanoisa as its provider; file_source_inputs declares compiler_support and
nanoisa. UTF-8 remains a host-native dependency and a private shared copy.
I do not suppress duplicate-symbol errors or discard arbitrary definitions.

My first new fixture used ambiguous implicit module identities; I retain that
setup error in initial-fixture.log. With explicit identities, the unchanged
baseline fails native linking on core_probe and incorrectly accepts a missing
provider (baseline.log). The corrected seven-method suite passes, including a
transitive diamond/cycle, an uncacheable provider shared with a direct import,
one generation per compilation, independent shared-library execution, and
missing-dependency refusal with preserved output. Existing header, long-link,
flag-overflow and same-module generation controls remain intact.

All seven methods also pass with ten compiler components instrumented by
ASan/UBSan (including module_builder, leak detection disabled). Three adjacent
native module-linking methods pass. Both actual source-input ABI methods pass
through the C seed, NanoVM, installed Stage 1/2 modules and sanitized C AOT;
these installed stages are existing producers, not refreshed bootstrap proof.
The File product bridge unit target passes.

My unchanged self-hosted compiler-product test passes in 553.159 seconds.
All eleven stages exit successfully with unchanged source inputs, original
stage deadlines and the ten-second shadow limit. I build the native C seed,
use it to compile my compiler to NanoISA, translate that module to strict C11,
and run the resulting native compiler through twelve capture, mutable-record
and scope acceptance methods. That compiler emits Hello, which executes with
exact expected output in both NanoVM and translated native C. I retain each
stage's output, commands, timings and source hashes in manifest.json and the
numbered logs. This is local compiler-product evidence; it does not establish
current raw Stage 1/2 equality or final Linux/Darwin release qualification.
