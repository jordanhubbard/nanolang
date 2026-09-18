# My checked map allocation contract

I track `task_bc7264a337074246953284ef892785d2` after the declared-tag
prerequisite. My heap constructor returns NULL if either allocation fails and
publishes no object or allocation statistics. HM_NEW must turn that result into
VM_ERR_MEMORY before putting a map on its operand stack.

I change my internal heap setter to return bool. Its arguments remain borrowed:
on success I retain inserted key/value edges; replacement retains the new value
before releasing the old value. On failure I retain neither input and preserve
prior contents, existing owners, and the map handle. Failed bucket allocation leaves the
old bucket allocation, capacity and contents unchanged. The general false-result
contract preserves contents; it does not promise an unchanged capacity if a
later slot search refuses after successful growth. I check doubling and
bucket byte-size representability before allocating. I use a wide intermediate
for the occupancy calculation. Existing-key replacement needs no allocation.

HM_SET first keeps its exact declared-tag check. If the checked setter fails,
I release its three consumed values and return VM_ERR_MEMORY with no result.
Other aliases remain valid; a later ordinary call can retry successfully when
allocation is available. This is failure propagation, not an unbounded retry
inside execution.

I inject deterministic failures only into the test-compiled heap allocation
calls. Controls cover constructor header/buckets, growth, valid replacement
under allocation refusal, unchanged prior contents, consumed reference cleanup,
and ordinary recovery. I retain normal map and VM gates and sanitizer evidence.
I do not alter lookup/delete policy, native map admission, collector scheduling,
or replay old failure artifacts. The internal C heap setter's bool status is
not a new serialized or foreign-module ABI.

## My acceptance evidence

At source `4fc09928` on main `44a13286`, `make test-nanovm
test-map-declared-tags` passes 274,493 ordinary VM checks, the complete existing
heap/stack/callback allocation harnesses, and seven paired map methods. The
paired source programs compile with normal shadows and execute on VM and
strict optimized standalone C with ASan/UBSan/LSan.

My new deterministic controls reject the map header allocation and initial
bucket allocation independently, then check an ordinary successful invocation.
They fill a valid map to its growth threshold, replace an existing entry while
bucket allocation is refused, and reject a new insertion both through the
borrowed heap API and through VM invocation. They check exact memory status,
no result/stack/frame residue, unchanged entry pointer/capacity/count/content
on failed allocation, input reference counts, successful later growth, and
final owner reclamation.

The dedicated sanitizer target instruments the allocation-shim heap itself,
changed VM dispatcher, cycle collector and value implementation. Other linked
compiler/runtime objects use their normal flags. GCC passes the complete
allocation harness with ASan/UBSan/LSan. I retain
`/tmp/nanolang-map-allocation-full.log` and
`/tmp/nanolang-map-allocation-sanitized.log`. These gates are ordinary corrected
source acceptance, not a full compiler fixed-point or new platform claim.

Clang also passes that complete instrumented allocation harness and all seven
paired methods (4.113 seconds); I retain
`/tmp/nanolang-map-allocation-clang.log`. The GCC paired run took 12.001 seconds.
No failure artifact replay or allocation-policy retry was needed.
