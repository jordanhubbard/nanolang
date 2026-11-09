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

I track the correction under #1121 and the matching roadmap checkbox. I retain the pre-correction baseline separately from the correction below. I require atomic allocation refusal,
unchanged existing owners/list/statistics, successful retry and the original
GC/environment/interpreted BSTRING sanitizer and platform controls before
closing the issue. Source timing, async, frame lifetime and release remain
separate open requirements.


## My correction

I make registry insertion report failure. `gc_alloc` registers the new payload
before linking it into the object list or charging statistics; on refusal I
free the unpublished header/payload allocation and return NULL. I preserve the
separate entry layout, public header, logical accounting and normal pointer
identity. This is not the retired private tail-storage optimization.

My permanent `tests/test_gc_allocation_failure.c` includes the actual GC
translation unit and controls only its malloc/free calls. It tracks real live
allocations and bytes without allocating tracker storage. It fails header and
registry allocation for ordinary, STRING and opaque constructors, preserves
two existing owners/list links/all five statistics, retries successfully,
checks retain/release, finalizers, repeated shutdown and reinitialization,
and requires zero tracked live storage. Neighboring runtime units use real
uninstrumented allocators; they are outside that injection domain.

My [final regression against original code](regression-baseline-final.json)
builds successfully and exits 1 at the required NULL assertion. My first
corrected-fixture attempt wrongly expected ordinary-release finalizers to
remain registered. I retain that failure in
`corrected-initial-finalizer-assumption.*`. Ordinary release removes registry
visibility before its callback; shutdown preserves visibility during its
callback. The corrected test checks both existing contracts, then still fails
the original allocation bug. I change no production finalizer ordering.

My [corrected local commands and hashes](corrected.json) pass all thirteen
existing GC controls, the allocation-failure matrix, interpreted BSTRING,
STRING bounds and the complete environment unit under ASan/UBSan, with the
existing Darwin leak-detector setting. Five actual native product hashes and
unchanged source inputs are retained. The existing Make sanitizer target now
runs the new matrix alongside the original GC controls.

Current full source/native/platform and Linux leak-detector execution remain
required. I do not close #1121 or claim release qualification from these local
controls. Original-source custody is refreshed only at a committed reviewed
source checkpoint, without widening metadata exceptions.
