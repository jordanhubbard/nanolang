# My C frontend owned-union checkpoint

I lower concrete generic and nongeneric unions through the same format-4
contract as my self-hosted producer. My registry owns nested argument trees,
places child instances before parents and derives resource flags from stored
payloads. Constructors preserve evaluation order; aliases, calls and returns
move whole owners. Selected complete patterns unpack exact field layouts;
ordinary/empty arms discharge their outer owner and reaching branches agree.

My original generic corpus and three original nongeneric controls pass through
both raw producers: 13 accepted sources verify and execute in NanoVM and native
C with ASan/UBSan/leak checks; nine refused sources preserve prior output. The
C frontend executes mandatory shadows before publishing these modules. The
unchanged three-compiler generic ownership/pattern/identity matrix passes all
44 methods. The broader C frontend selected-variant probe matches 15 of 16
acceptance expectations; its unchanged exactly-once scalar-global case still
refuses and remains tracked separately.

I instrument the C codegen and checker translation units with Homebrew LLVM
ASan/UBSan and enable leak detection. All 22 corpus cases pass. Other linked
translation units remain ordinary objects; this is not a fully instrumented
compiler qualification. The first instrumented run exposed 45 leaked bytes in
five inherited symbol type names. I retain that terminal and release replaced
selected-binding, inferred-binding and parameter names before rerunning.

The new match root initially retained only its final arm's lexical name. I
retain that one-interval observation; my regression now requires both disjoint
payload intervals and the existing local-name codec's canonical round trips.

The complete scalar-union source and source-borrow regression processes remain
running at this checkpoint. They rebuild self-hosted emitters; I do not infer
their outcome from the focused passes. Full integrated platform/sanitizer and
final-source fixed-point qualification remain incomplete. PR #522 stays draft.

`checkpoint-provenance.json` pins implementation, test and executable hashes;
`checkpoint-logs.json` hashes the uncompressed retained logs. From this checkout,
`PYTHONPATH=. python3 docs/evidence/pr522-implementation-2026-09-23/c-frontend/instrument_frontend.py`
replays the scoped check using the captured Darwin toolchain commands and
current built dependency objects. It writes scratch outputs beneath `/tmp`.

# My historical C frontend parity baseline

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
