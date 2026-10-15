# My capture qualification disk-space recovery

My ordinary isolated checkout of `ae92c0488` exits 128 while writing tracked
`docs/evidence` files: the filesystem reports no space left. The attempted runner
then exits 2 because preparation never created its script. No compiler-product
test ran in that attempt. I record the exact source pin and observed disk usage
in `checkout-failure.json` and remove only that incomplete disposable checkout.

I retain my older qualification clones and their host-library paths. I prepare a
new independent Git checkout by APFS-cloning all 61,352 tracked files from the
clean committed source, using a separate index and shared object database. I
verify an empty Git status at the exact source pin before starting the same full
compiler-product command. `cow-checkout.json` records the preparation; the two
Python scripts reproduce its filesystem and test setup.

The retry is running in `/private/tmp/nanolang-capture-ae92c0488-cow` with evidence
under `/private/tmp/nanolang-capture-ae92c0488-cow-evidence-20261008`. This archive
records preparation only, not a passing terminal. I must retain the terminal
before claiming qualification.

A separate `make -j2 test-bytecode-shadows test-parser-parenthesized` run overlaps
the disk-full interval. Its first 40 shadow tests and 56 cache-publication tests
pass; the Linux-only cache suite skips four methods on Darwin. Its still-running
source-snapshot phase reports errors. I must inspect its final diagnostics and
rerun the failed qualification after the resource condition is resolved; I do
not attribute every failure to disk pressure without checking its evidence.

My completed `9790d9a91` checkout now shares identical evidence-file storage
with the real checkout through APFS clonefile. I compare tracked blob identities
and verify source bytes before replacing each duplicate, then verify the old
checkout remains clean. `dedup.json` records the completed operation and disk
recovery; host libraries and their paths remain intact. My clean retry at
`ae92c0488` now passes all 109 compiler-product methods; its terminal archive is
`../compiler-product-ae92c0488`. The separate source-snapshot terminal remains
pending and is not covered by that successful gate.
