# I map empty owned records before reconstruction admission

I record design/baseline ownership under #999 from765a698d4. My authority is
the existing required ownership-through-reconstruction row9538 and the exact
accepted `OwnedRecordPatterns.test_empty_resource_pattern` source.
`test_source_borrow_emission.py:test_temporary_owners_restore_original_pattern_sources`
passes its PREFIX plus unchanged case through the paired ordinary/shadow producer
gate. I retain the PREFIX rather than invent a smaller source.

```nano
resource struct Empty {}
fn main() -> int { let value: Empty = Empty {} let Empty {} = value return 0 }
shadow main { assert (== (main) 0) }
```

The existing facts query rejects zero field_count, and the owned analyzer also
requires positive field count. This source/runtime versus reconstruction
mismatch needs an actual baseline before admission. No new field type, wrapper
authority, array or arbitrary empty ordinary struct is implied.

I must preserve exact resource declaration/layout identity even when two records
have no fields. No field data does not erase ownership: construction creates one
owned value, move transfers it once, and empty destructuring consumes it once.
There must be no duplicate use, skipped consumption, escaping owner or invented
scalar placeholder. Empty tuples must not collapse distinct nominal layouts or
be confused with absent authority. Existing verifier/lifetime and output limits
stay in force. A future implementation must map zero-field C/Nano representation
and original/regenerated verifier acceptance explicitly, without assuming an
empty C struct is portable.

The bounded baseline uses current seed and both installed stages for ordinary
source, and an independently pinned original-shadow driver where applicable.
I retain exact source, function/layout/local counts, module dumps, original
verifier/VM outcomes, facts/refusal status and atomic prior-output preservation.
Original shadow input provenance remains separate from regenerated output and
from any consistency witness. No native/frontend rebuild or production edit
is part of this design checkpoint. The baseline may reveal more than the
nonempty-layout guard; I stop and report those concrete facts before expanding.
