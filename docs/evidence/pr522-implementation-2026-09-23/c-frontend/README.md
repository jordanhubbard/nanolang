# My C frontend parity baseline

At source checkpoint `4d23f3cb2`, I run the unchanged source strings from
`tests/test_generic_selected_ownership.py` through the freshly built
`bin/nano_virt source.nano --emit-nvm -o out.nvm`. I initialize the output to
`prior module` before each case. `baseline.json` retains every exit status,
stdout, stderr and prior-output comparison; `provenance.json` pins my producer,
source corpus and executable.

All ten accepted cases stop before shadow publication with `scalar concrete
union arguments`. All nine negative cases fail in checked source ownership
and preserve prior output. The guarded diagnostic method invokes its original
three-compiler matrix directly and is not duplicated in this probe. A successful
negative here does not exercise the C producer's transfer checks.

My next implementation order is concrete instance registration and retained
child layouts, whole-owner construction/alias/call/return transport, then selected
match extraction and joins. Registration must own any synthesized nested type
arguments: `resolve_union_payload_type_info` returns temporary owned trees.
Resource classification must follow stored payloads rather than unused type
arguments. Complete publication still requires mandatory shadows, verification,
VM/native execution and preserved negative outputs for this same corpus.

This baseline does not qualify C frontend acceptance or release readiness.
