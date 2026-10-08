# I retain the recovered source-snapshot terminal

At clean f0f6a0c62b9a3fb53251f987a492f3de9f7ab514 I run all 127
source-snapshot methods in 5244.900 seconds: two subcases fail and 19 skip.
The runner exits 1 after 5252.277 seconds with unchanged HEAD, source and
probe hashes. Final free space is 67,804,532,736 bytes.

The external/shared assembler search-order case refuses tool-hash evidence
at its deadline. The unknown-fragment fallback case with shell parameter
expansion fails original module compilation. These diagnostics do not prove
either root cause. I preserve the full run and require targeted diagnosis,
correction and complete applicable qualification in [#980](https://github.com/jordanhubbard/nanolang/issues/980).
