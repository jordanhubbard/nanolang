# My scalar-global declaration and private-flow checkpoint

I retain exact scalar-global slot tags in ownership format 4, extension kind 3,
revision 1. My [contract](../../../NANOISA_OWNED_SCALAR_GLOBALS.md) fixes the
encoding, limits, initialization rules and remaining VM/native/source work.
The query validates the complete ownership payload before publishing any tag
or count. Invalid tags, reserved bytes, versions, duplicate/unknown kinds,
truncation, extra bytes and insufficient output capacity refuse without changing
query outputs. Both ordinary and maximum-sized declarations round-trip.

My private affine analysis tracks definite global initialization alongside local
and stack state. Branch and loop joins intersect initialized slots. Entry cannot
read an uninitialized slot, call a helper before every declared slot is ready,
or exit with incomplete initialization. Helpers rely on that checked caller
precondition. Stores require the declared scalar tag and reject owner tokens
and rooted observations. Global bits participate in the bounded visit budget.

Twenty-one positive/refusal flow cases cover all five scalar tags, multiple
slots, branch/loop joins, owner/observation stores and helper calls. Normal
transport/flow tests pass 105,066 assertions; allocation-failure tests pass
131,242. These counts include byte-by-byte unchanged-output checks, not that
many independent source programs. The same checks pass scoped ASan/UBSan with
leak detection, instrumenting ownership contracts, affine state, affine bytecode,
verifier and fixtures. Other linked dependencies remain ordinary objects.
`instrument_contracts.py` reproduces this check from current built objects.

The existing affine bytecode checks pass 837 normal and 1,199 allocation-fault
assertions, all 33 schema methods pass, and the complete native translator
regression passes all 2,428 assertions. My first new fixture used textual
aggregate/Boolean operands where the assembler requires numbers; I retain that
terminal and correct only those test operands.

Public owned execution with scalar-global declarations remains refused. Private
analysis does not qualify VM/native initialization, storage or cleanup, and
neither source producer emits this extension yet. The original exactly-once
counter therefore remains an unmet source acceptance requirement. PR #522 and
release stay held. `provenance.json` and `logs.json` pin source and terminal hashes.
