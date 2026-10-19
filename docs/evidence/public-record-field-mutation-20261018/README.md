# I qualify bounded public record field replacement

I ran `make -j2 test-record-array-public` on Darwin with Apple Clang; the complete gate passed in 216.359 seconds. My exact source hashes are retained in `tested-inputs.json`. My strict gate passed 222 Clang and 222 GCC translation-unit builds.

I qualify all five flat element tags through public verification, VM execution, canonical assembly and serialized v2 transport. My mutation fixtures exercise initializer globals, nested records, local/global/call aliases, repeated field replacement, retained old GET values, boxed global arrays in constructors and replacements, and wrong-field refusal before execution state changes.

I execute 34 successful native programs/probes and 30 deliberately failing assertion-prefix programs at O0/O2 with ASan/UBSan. Successful mutation programs force collection, keep global record/array owners alive, clear the actual global roots, and require zero canonical/array/live-byte owners before disposal. The fatal native prefixes assert the expected assertion abort and no sanitizer diagnostic; they do not establish native in-process recovery.

I inject transient and persistent allocation failure at 472 flow and 1845 mutation execution positions per tag: 23,170 total injected prefixes. Each separately bounded tag/fixture invocation preserves the complete corpus, requires VM failure cleanup and zero tracked ownership after destruction, restores allocation, and reruns successfully. Setup, uninstrumented common providers, transport allocation failures and native allocation-prefix recovery remain outside this receipt.

I preserve earlier test-wrapper failures: missing initializer invocation, followed by the string-array owner-list spelling. Both were corrected without production changes or weaker lifetime assertions. The final complete gate above includes the corrected wrappers. Earlier classifier/emission shape and boxed-constructor failures are recorded in my ROADMAP.

I still require fresh combined compiler generations and exact-head hosted qualification. This receipt does not complete paired source publication, generic layouts, or the broader 5.1 scope.
