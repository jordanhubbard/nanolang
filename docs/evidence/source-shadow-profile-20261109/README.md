# My source-shadow profiling attempt

I retrieve Apple comparison artifact 13639517823 from run 42534611623 and verify
its complete ZIP CRC inventory and original C-provider SHA256 against the retained
before/after manifests. My `plan.json` records exact bytes and source provenance.
This actual arm64 image has the existing libSystem, OpenSSL 4 and libffi dependency
paths and loads on this machine. My current C `src/` and Makefile.gnu bytes match
its 9baeba459 build source. I do not infer qualification of the relocated runtime
root or current NanoLang source inputs from that C-source comparison.

I make the one declared diagnostic attempt with unchanged shadow selection and
supervisor. Setting CC=/usr/bin/false also affects imported-module compilation
before the shadow phase. The compiler refuses `compiler_schema.nano` creation;
zero shadows start, no sample is attempted and no native product is published.
I retain exact commands, diagnostics, source/provider before/after hashes and the
measurement driver. I do not claim profiling or source-creation success. I do
not repeat the attempt with the same limitation. Local full builds remain held
while available disk headroom is below policy.

My `hosted-body-ranking.json` instead analyzes already retained original trace
text at exact 869f11634. All three workers complete the same ordered 1,284 shadow
bodies with zero recorded assertion failures. The largest individual body is
nisa_emit_function (Apple 0.428718, GCC 0.402039, Clang 0.318045 seconds). The next
bodies include nisa_emit_expected_expr, parser block recursion and owner-array
controls. These historical intervals are not CPU samples, exclude surrounding
shadow work, and do not qualify my current candidate or establish a cause for
Apple's ten-second deadline. I preserve every assertion and selection.

I retain a 12 MB local copy of the observed C-provider image under
`.test_output/source-shadow-profile/`. The downloaded ZIP is temporary and can be
removed after these hash/manifests are committed; the exact CI artifact remains
the retrieval source. No native SDK or lifecycle admission follows from this
probe. The actual CPU profile and original platform/budget acceptance remain
open under #1065.
