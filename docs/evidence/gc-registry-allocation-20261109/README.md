# My registry-allocation failure baseline

I fail exactly the second malloc in my actual production `gc.c`: payload/header
allocation succeeds and registry-entry allocation fails. My
[probe source](probe.c), [commands and source/product hashes](baseline.json),
[stdout](run.stdout) and [stderr](run.stderr) retain the actual result.
The ASan/UBSan build succeeds and the probe exits 2: I return an owner, report
it unmanaged, and retain one object after ordinary release. Shutdown then
reclaims the owner; the probe never dereferences a released payload.

I wrap only malloc in the included production GC translation unit. Neighboring
runtime units use their real allocators. I keep the existing Darwin leak-detector
setting and do not claim Linux, current hosted qualification or a fix.
The recorded source revision 869f11634 has the same GC bytes as the branch's
7a709ee6b base; actual file SHA256 values are retained independently.

I track the correction under #1121 and the matching roadmap checkbox. I have
not changed production allocation behavior. I require atomic allocation refusal,
unchanged existing owners/list/statistics, successful retry and the original
GC/environment/interpreted BSTRING sanitizer and platform controls before
closing the issue. Source timing, async, frame lifetime and release remain
separate open requirements.
