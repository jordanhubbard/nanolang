# I reconstruct bounded local resource transfers

I advance required ownership reconstruction under #999 and rows 9538/10409 with
one closed, single-entry module: resource records whose ordered fields are exact
INT or BOOL, local construction, move, consuming unpack and scalar return.
I retain every numeric nominal layout identity and resource declaration. Two
same-shaped layouts remain distinct. I do not claim the wider owned-call,
borrow, generic, nested, union, array, effect or control-flow requirement complete.

I require successful current `nvm_verify` executable admission before extracting
facts. Affine instruction analysis is additional consistency evidence, never a
replacement for execution admission; advisory frontend facts grant no authority.
I expose the exact retained layouts and local declarations and preserve complete
instruction order. My owned region statements are never duplicated, discarded
or folded through scalar temporary optimization.

My C output uses concrete uniquely named structs and explicit one-use transfer
temporaries. My Nano output uses `resource struct` and consuming `let Type {
field_names } = owner` syntax. That syntax does not support field aliases;
repeated same-name patterns are supported with distinct lexical slots. I snapshot
scalar results immediately after each pattern. Both outputs consume one checked region tree and contain no embedded VM, goto or
hidden bytecode dispatcher. Deterministic numeric names preserve layout identity
without claiming original source spelling recovery.

I admit only the actual producer's exact ephemeral shared-field read sequence
`REGION_BEGIN/BORROW_LOCAL_SHARED/REF_GET/REGION_END` on a live exact local
owner. I represent it as a read statement, require matching reference index and
valid scalar field, and forbid escape or overlapping/misnested regions. I retain
terminal consuming `OWN_UNPACK_LOCAL`. I initially refuse branches, loops, calls,
other references, globals, imports, nested records and unsupported instructions in this resource route. I retain the
existing scalar route unchanged. All refusals preserve a prior output file.

Acceptance requires actual C-hosted source producer constructor/move/destructure
fixtures, same-shaped distinct nominal records, field ordering and once-only
scalar computations. I compare original NanoVM with reconstructed native C and
reconstructed NanoVM/native execution, and inspect regenerated resource layout
correspondence. Missing/mutated authority, wrong nominal identities, duplicate or
lost owners and use after move must refuse. Fresh Stage1/Stage2, Linux/Darwin and
broader ownership paths remain separate unfinished qualification.

My first source probes disproved two artificial assumptions: a standalone block
is not accepted syntax, and `if true` does not erase branch ownership obligations.
Repeated patterns are accepted. Real source destructuring emits ephemeral reads
followed by a delayed consuming unpack, so an owned-opcode-only allowlist would
not qualify the real producer. I preserve those failed probe logs and expand
only the exact observed read sequence, without broad borrowed reconstruction.
