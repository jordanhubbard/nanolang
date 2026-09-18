# Executable scalar region reconstruction

I implement MAC `task_15c7bd012efc44ffab908df7cd66b71e` as a bounded child of
`task_4bd034f6029b7458201db74e2c3aeb32`. I read one verified `.nvm`, without its
original source, recover a typed region tree, and emit both C and NanoLang that
compute directly. I do not embed bytecode, a program counter, an opcode loop,
a VM library, or goto as a substitute for structured control.

## Admission and representation

I use my existing loader, `nvm_verify` and `vm_decode_function` to extract exact
function indices, signatures, instruction boundaries and branch targets. A
small host facts tool feeds a reusable host region analyzer; this JSON is an
internal tool interface, not a new portable module format or trusted metadata.
Both output surfaces consume the same checked region tree.

I initially admit at most 32 functions, 128 locals per function, 4,096 decoded
instructions per module and region nesting depth 32. Expanded scalar expressions
have at most 4,096 nodes and depth 128; shared stack operands cannot cause
unbounded textual expansion. Functions have no upvalues,
int/bool parameters with retained exact tags, and one explicit int/bool result.
I require the explicit HAS_MAIN entry flag; the entry has no parameters and
returns int. An index-zero function alone does not establish an entry. Direct calls form an acyclic graph.
I refuse unknown signatures rather than defaulting them to int.

My opcode whitelist is NOP, PUSH_I64, PUSH_BOOL, LOAD_LOCAL, STORE_LOCAL, DUP,
POP, SWAP, signed I64 comparisons, BOOL_AND/OR/NOT, CALL, JMP, JMP_FALSE,
JMP_TRUE and RET. The exact integer child now also admits I64_ADD, I64_SUB and
I64_NEG through total cross-language helpers; see [the arithmetic contract](NANOISA_RECONSTRUCT_INTEGER_ADD.md).
I64_MUL now uses the same typed tree with [bounded total multiplication](NANOISA_RECONSTRUCT_INTEGER_MUL.md).
I64_DIV_S and I64_REM_S use [guarded signed division and remainder](NANOISA_RECONSTRUCT_INTEGER_DIV.md), including defined zero/overflow results.
I64_SHL, I64_SHR_S and I64_SHR_U use [masked-count portable helpers](NANOISA_RECONSTRUCT_INTEGER_SHIFTS.md), keeping arithmetic and logical right shift distinct.
I64_AND/OR/XOR/INVERT preserve [exact64-bit patterns](NANOISA_RECONSTRUCT_INTEGER_BITWISE.md) through unsigned C and bounded NanoLang helpers.
I64_LT_U/LE_U/GT_U/GE_U retain [unsigned bit-pattern ordering and boolean results](NANOISA_RECONSTRUCT_UNSIGNED_COMPARISONS.md).
Other arithmetic, generic tagged comparisons, implicit returns, multiple returns,
tail calls and all other operations remain outside this slice.

I infer exact scalar local types from declared parameters and stores, reject
mixed-type reuse and uninitialized reads, and require an empty operand stack at
region joins and loop backedges. I preserve evaluation order using typed
source temporaries; a later store cannot change an already loaded operand.

I recognize straight-line blocks, forward if/else diamonds and pretest loops
with one condition exit and a final backedge to the header. Nested regions are
allowed within the same bounded grammar. I reject crossing jumps, multiple loop
entries, unsupported exits, irreducible graphs and stack-valued joins before
publishing output. Each decoded instruction must belong to the accepted tree;
I do not hide unhandled reachable code behind a fallback interpreter.

I refuse imports, module links, globals, aggregates, retained ownership/passive
contracts and effects. This is a closed executable reconstruction slice, not
host ABI or general compute-profile completion. Existing production `nvm2c`
remains independent, including its larger supported opcode set.

## Names and output

Function indices and local slots remain authoritative identities. I derive
collision-free target identifiers from those identities, optionally adding
sanitized original spelling. Advisory names cannot create aliases, keywords,
code or types. Reused slot names do not imply multiple runtime storage locations.
Absent or unusable advisory names use deterministic generated names. DEBUG facts
remain advisory and do not alter graph admission or execution.

I expose a separate `nvm2hl --language c|nano input.nvm -o output` host tool.
I finish analysis and emission before replacing the output file. Failed
admission preserves an existing output. No product compiler cutover occurs.

## Acceptance and limits

From the same retained module, I execute NanoVM, compiled C and reconstructed
NanoLang compiled by C seed and selfhosted compilers. Fixtures cover both branch
outcomes, loop zero/one/multiple iterations through scalar state transitions,
nested regions, scalar calls, shadowed source spelling and absent metadata.
I inspect generated executable syntax for real if/while/return and the absence
of interpreter/goto fallback. Negative controls retain prior output for unknown
signatures, mixed local types, remaining unsupported arithmetic and graph forms.

The test harness supplies independently specified shadow assertions for generated
NanoLang functions. Those assertions are validation-only: I do not reconstruct
original mandatory tests that the input module never retained. I state that
limitation without adding a new release requirement. Full high-level roundtrip,
other producer families, general graph recovery, remaining arithmetic, host imports and
frontend facts remain required parent obligations.
I publish sufficient/insufficient findings only for the exact tested subset.

I also reconstruct exact int/bool `CAST_INT`, `CAST_BOOL`, `AND`, `OR` and `NOT` under [my bounded truthiness contract](NANOISA_RECONSTRUCT_TRUTHINESS.md). Evaluated operand snapshots precede boolean combination; other tags remain refused.

I reconstruct exact integer `I64_MUL_WIDE_S` and `I64_MUL_WIDE_U` with immutable low/high word snapshots under [my portable limb contract](NANOISA_RECONSTRUCT_WIDE_MULTIPLY.md). I keep the blocked carry/borrow compiler acceptance separate.

I also reconstruct generic ADD/SUB/MUL/DIV/MOD/NEG when every operand is statically INT, using the unchanged total helpers in [my exact-INT contract](NANOISA_RECONSTRUCT_GENERIC_INTEGER.md). This does not admit dynamic numeric promotion or other tags.
