# My deferred cycle-root accounting

During collection, I detach the old suspect buffer and let releases queue new
suspects for the next pass. My old candidate graph can still reach a newly
queued root. Trial deletion then discounts that root's outgoing references,
but its buffered flag keeps white collection from detaching its fields. A later
ordinary release would discount those intact edges again.

I now restore trial-touched (`GRAY` or `WHITE`) fresh-queue subgraphs before
collecting old white objects. I finish all restorations before normalizing any
fresh root to `PURPLE` (positive count) or `BLACK` (zero count). This preserves
`scan_black`'s visited marker across overlapping subgraphs. Untouched fresh
purple roots need no restoration. Old roots reached by restoration survive this
pass, and deferred zero-count roots retain balanced child ownership until their
normal release in the next pass.

I ran the existing positive VM lifecycle and cycle-collection suite on this
repair: `make -j8 test-nanovm` passes 272,579 checks, plus the existing field
allocation and stack reserve failure/recovery checks. I added no minimized
failure input and ran no invalid-access diagnostic. This is a static accounting
repair with existing-suite evidence; I have not established that it resolves
the complete compiler shadow assertion under
`task_fdf43892a1104b1facddc2553af390af`.

Task: `task_72433f501ddd4736a45e6244c44ae4fa`. Base: `8e329159`.
Local log: `/tmp/nanolang-vm-deferred-cycles-tests.log`.
