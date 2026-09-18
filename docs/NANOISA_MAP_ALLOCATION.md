# My checked map allocation contract

I track `task_bc7264a337074246953284ef892785d2` after the declared-tag
prerequisite. My heap constructor returns NULL if either allocation fails and
publishes no object or allocation statistics. HM_NEW must turn that result into
VM_ERR_MEMORY before putting a map on its operand stack.

I change my internal heap setter to return bool. Its arguments remain borrowed:
on success I retain inserted key/value edges; replacement retains the new value
before releasing the old value. On failure I retain neither input and preserve
prior contents, existing owners, and the map handle. Failed growth leaves the
old bucket allocation, capacity and contents unchanged. I check doubling and
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
