# My scalar globals in owned value graphs

MAC `task_6fa28259d6c676a11406fd124a7d28f9` tracks this dependency of PR #522.
My unchanged `test_scrutinee_call_evaluates_once` increments a mutable global
inside a union-producing function. Both native stages and the C bytecode
frontend must preserve that effect in production and mandatory shadows.
I do not replace the counter with a constant or remove its assertion.

## My retained declaration

I add mandatory-understanding ownership extension kind 3, revision 1,
`SCALAR_GLOBALS`, within ownership format 4. Earlier ownership versions retain
their existing profiles. Kinds remain unique and ascending. The payload is a
u32 count followed by that many four-byte descriptors: one scalar tag byte and
three zero reserved bytes. Count is 1 through 256, matching my bounded owned
source profile. Slot order is declaration order. I admit INT, U8, BOOL, FLOAT
and STRING tags; no aggregate, owner, reference, function or unresolved tag is
a scalar-global declaration. Empty extension payloads and trailing bytes refuse.
I do not combine this contract with the private managed-array profile.

A validated declaration supplies exact slot types, not executable authority.
My query validates the complete ownership payload before publishing any result;
malformed declarations and insufficient output capacity leave outputs unchanged.
A valid module without this extension has zero declared scalar globals.

## My initialization and transfer rules

I preserve source-order initialization in the entry function, including the
synthetic shadow entry. Initializers must belong to my checked scalar expression
profile. Each slot must be initialized before a read. Every declared slot must
be initialized before entry calls an owned helper or returns successfully.
Definite initialization meets by intersection across reaching entry branches.
My acyclic helper graph cannot call entry and may assume initialization only
because all entry-to-helper calls establish that precondition.

`LOAD_GLOBAL` copies only its declared scalar value. `STORE_GLOBAL` consumes a
matching scalar stack value and never an owned token or rooted observation.
Wrong slot, wrong tag, missing initialization, incomplete branch initialization
and a call before initialization refuse. Mutable source bindings retain checked
assignment rules and lexical locals continue to shadow globals.

My existing VM global storage must preserve these admitted operations, including
scalar STRING retention and release. Generated native code must preserve the
same tags, initialization, mutations and cleanup. Declaration transport alone
must not enable instructions before verifier and runtime/native support agree.

## My acceptance

I first test complete declaration transport, malformed descriptors, unknown and
duplicate extensions, truncation, version boundaries and failure-atomic queries.
Then I test raw VM/native accepted effects and refused entry/call/join/owner
transfers. I retain allocation and ASan/UBSan/leak controls. Finally both source
producers and fresh native stages must pass the original exactly-once program
with its original shadows, plus prior-output refusals and existing ownership,
scalar-union and source-borrow regressions. Full release gates remain required.
