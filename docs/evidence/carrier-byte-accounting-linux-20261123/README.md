# I distinguish an allocation diagnostic from my ordinary gate

I retain this bounded Linux ARM64 observation at source `8704f4ef57`, following
[my original receipt](../carrier-string-scopes-linux-20261123/README.md).
I change no product source, source fixture, shadow, allocation threshold or
qualification deadline. I keep #1213, default ownership and release acceptance
open.

My external linker wrappers count requested and usable outstanding storage from
malloc/calloc/realloc/free/strdup/strndup calls made by linked provider objects.
A fixed static table stores observations without allocating diagnostic metadata.
This does not intercept every libc internal allocation. Four unobserved frees
occur before the first sample; their count stays four. Duplicate/overflow counts
stay zero. The observer links the exact retained provider objects and unchanged
complete source/shadow workload; all provider object bytes are archived and
matched to the preceding receipt's identities.

| Observation | After 64 calls | After 128 calls | After teardown |
|---|---:|---:|---:|
| Instrumented requested bytes | 130169 | 130169 | 0 |
| Instrumented usable bytes | 143152 | 143152 | 0 |
| Instrumented live allocations | 1042 | 1042 | 0 |

The instrumented native observer also reports zero second-batch growth. Its
positive control remains 2101904 bytes and all original ten shadows, 128 calls,
carrier/identity guards and unchanged 1024-byte criterion remain intact. Changing
the executable changed the observation; this does not qualify the original
ordinary observer, whose retained +1776-byte failure remains authoritative.

One separate allocator-axis control executes the **unchanged original binary**
with `GLIBC_TUNABLES=glibc.malloc.tcache_count=0`. It reports zero second-batch
growth, a 2101904-byte positive control and all 128 calls. This demonstrates
sensitivity to this allocator setting in the retained workload. It neither
identifies the exact original retained chunks nor proves a historical cause.
I do not install this setting, raise the allowance or claim an ordinary-default
pass. No unchanged retry or compiler/source/platform acceptance follows.

I retain an initial diagnostic build refusal caused by duplicate `_GNU_SOURCE`
under `-Werror`; I removed only the redundant diagnostic define before building.
All jobs are terminal. Puck source ownership, frozen roots, audits/stashes and
NAS/TCC/migration pauses remain unchanged. I hold no native worker reservation.

Run `python3 docs/evidence/carrier-byte-accounting-linux-20261123/read_archive.py`
from this Git checkout. I verify all 134 record hashes, the prior manifest and
provider identities, exact observer insertions, original threshold and failed
output, samples/controls, actual ARM64 diagnostic executable and explicit
nonacceptance flags. These are diagnostic observations, not a whole-release gate.
