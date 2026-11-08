# My STRING self-assignment ownership correction

Under #1112 I retain a direct C baseline that creates one managed STRING binding, assigns its exact Value back and observes `gc_is_managed` change from one to zero. The probe exits two without dereferencing or freeing the dangling value. The added deterministic environment unit also fails its managed-pointer assertion before the correction.

I preserve the binding's ownership when the old and new STRING pointers are identical, while still copying the supplied Value metadata. I keep the reference count unchanged and retain the existing struct-copy and different-pointer alias behavior. The complete original environment/index/allocation-failure unit now passes, including managed and ordinary allocated self-assignment, aliased replacement, exact managed reference counts, Value return-target metadata and final release.

A separate ASan/UBSan build instruments `env.c` and the C unit; unchanged linked common objects retain their ordinary flags. It passes with no sanitizer failure. This is not a full sanitizer bootstrap. An actual language control also executes interpreter shadows and a generated native program, exercising lone and aliased STRING self-assignment, replacement, parameter copies and caller preservation. Source/provider hashes and before/after native product hashes independently match, and the actual product is Mach-O.

The private ZIP passes CRC and retains 36 file hashes and thirty exact texts, the baseline probe, ordinary/sanitized unit products and the generated language product. I retain the exact pre-fix source and patch. Complete hosted/platform integration and draft review #1111 remain open; the separate checker deadline optimization is not claimed by this correctness fix.
