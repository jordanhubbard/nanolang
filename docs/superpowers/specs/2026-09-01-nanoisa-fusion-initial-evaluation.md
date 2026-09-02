# NanoISA Fusion Candidates: An Initial Evaluation

Roadmap 4.0, Phase 12, Execution architecture.

I decode each function once and dispatch predecoded instructions
(`src/nanovm/vm_dispatch.c`). That predecoded IR is where a fusion would live: a
fusion replaces a fixed sequence of predecoded instructions with a single
private dispatch handler, without adding any portable opcode. This document is
the *initial* evaluation the roadmap asks for. It names the five candidates,
locates each one in the code I actually generate, states what a fused handler
would remove, and records the measurement I would demand before I accept any of
them. It does not accept a fusion. Acceptance is a separate roadmap item, and it
is governed by `docs/NANOISA_OPTIMIZATION_POLICY.md`, not by this evaluation.

## What a fusion is here, and what it is not

A fusion is a *private dispatch optimization*. It is invisible to the portable
instruction set: `spec/nanoisa.yaml` gains no opcode, the assembler and
disassembler gain no mnemonic, and the verifier still checks the original
instructions. The predecoder recognizes a maintained sequence and rewrites those
predecoded slots to run one handler that does the combined work. If a workload
never contains the sequence, nothing changes. This is deliberately different from
a *superinstruction* exposed as an opcode — I refuse to leak frontend bookkeeping
into portable opcodes, and the roadmap keeps that as its own item.

A fusion is legitimate only when it removes real per-instruction overhead:
dispatch bookkeeping, a stack push immediately consumed by the next instruction,
or a redundant boundary lookup. It is illegitimate when it only merges two
handlers whose combined body is the same work in one `case` label — that is a
larger switch, not a faster one.

## The five candidates

I generate all five sequences today. The line references are to the lowering in
`src/nanovirt/codegen.c` and the opcodes in `src/nanoisa/isa.h`.

### 1. Local-field load

**Sequence.** `LOAD_LOCAL slot` followed by a field/element read on the loaded
aggregate — `STRUCT_GET field`, `UNION_FIELD field`, `TUPLE_GET index`, or
`AGG_GET field`. I emit exactly `LOAD_LOCAL; ...GET` wherever code reads a field
of a local, and the map/filter/reduce lowering does this on every iteration
(`LOAD_LOCAL src_slot; LOAD_LOCAL idx_slot; ARR_GET`).

**What a fusion removes.** One dispatch turn and one operand-stack push/pop pair:
the loaded aggregate is pushed only to be popped by the getter on the next
instruction. A `LOAD_LOCAL_FIELD slot, field` handler reads the local, indexes
the field, and pushes the field value once.

**Risk.** The getter opcodes are heterogeneous (struct, union, tuple, array,
generic aggregate) with different bounds and ownership rules. A fusion must
either specialize per getter or handle a tagged aggregate at runtime, and the
retain/release effect of the read must be preserved exactly. The gain is one
stack round-trip per hit; whether that clears run-to-run noise is the open
question.

### 2. Local increment

**Sequence.** `LOAD_LOCAL slot; PUSH_* k; ADD; STORE_LOCAL slot` — the canonical
loop-counter step. Every counted loop I lower ends its body with this exact
four-instruction update (`codegen.c` loop tails: `LOAD_LOCAL i; PUSH 1; ADD;
STORE_LOCAL i`).

**What a fusion removes.** Three dispatch turns and three stack round-trips
collapse into one `INC_LOCAL slot, k` handler that adds a small immediate to a
local in place. This is the densest, most repetitive candidate: it fires once per
loop iteration in recursion, `sum_loop`, `primes`, `fib_iter`, and the Forth
inner interpreter.

**Risk.** NanoLang integers are typed values on a hybrid stack; the in-place add
must honor overflow behavior and the value representation exactly as `ADD` does,
and must reject the fusion when the constant is not an integer immediate the
predecoder can fold. The upside is the largest of the five, which makes it the
first candidate I would measure.

### 3. Compare-branch

**Sequence.** A comparison (`EQ`, `NE`, `LT`, `LE`, `GT`, `GE`) immediately
followed by `JMP_TRUE` or `JMP_FALSE`. Every loop guard and every `if` on a
relational test lowers to `...; LT; JMP_FALSE target` (`codegen.c` loop headers:
`LOAD_LOCAL i; LOAD_LOCAL n; LT; JMP_FALSE`).

**What a fusion removes.** The boolean the comparison pushes is consumed by the
branch on the very next instruction; a fused `BR_LT target` handler compares and
branches without materializing the boolean or taking a second dispatch turn.
Because the branch target is already resolved to a predecoded index in
`vm_dispatch.c`, the fused handler can jump directly.

**Risk.** Six comparisons times two branch senses is twelve handlers, or one
parameterized handler. The fusion must preserve the exact truthiness `JMP_TRUE`/
`JMP_FALSE` use and must not fire when the boolean is also consumed elsewhere
(for example a `DUP` between the compare and the branch). The predecoder already
has adjacency and boundary information, so detection is cheap; the payoff is one
turn plus one stack slot per guard.

### 4. Union-tag branch

**Sequence.** Reading a union's tag and branching on it. Match lowering uses
`MATCH_TAG variant, offset` (which already tests a tag and branches in one
opcode), and hand-written dispatch uses `UNION_TAG` followed by a compare and a
conditional branch. The candidate is the `UNION_TAG; PUSH variant; EQ; JMP_*`
chain, and the question of whether repeated `MATCH_TAG` arms over one scrutinee
should share a single decoded tag.

**What a fusion removes.** For the `UNION_TAG; ...; EQ; JMP` form, the same
compare-branch saving as candidate 3 plus the tag extraction. For a match with
many arms, decoding the scrutinee's tag once and running a small jump table over
the arms would remove one tag read and one branch per arm after the first.

**Risk.** `MATCH_TAG` already fuses tag-test-and-branch into a single portable
opcode, so much of the naive win is *already captured by the ISA*. The remaining
opportunity is a decoded jump table across sibling `MATCH_TAG` arms, which is a
predecoder pattern over several instructions, not a two-instruction peephole. It
is the most speculative candidate and depends on how often real matches have
enough arms to beat sequential `MATCH_TAG`.

### 5. Tail-call fusion

**Sequence.** Argument setup immediately followed by `TAIL_CALL index`. I already
lower direct tail calls and replace the frame in place, with verifier and runtime
signature checks (the completed roadmap item on tail-call lowering). The fusion
question is whether the argument-store-then-`TAIL_CALL` prologue can be fused with
the frame replacement so the outgoing arguments are written directly into the
reused frame slots.

**What a fusion removes.** The copy of already-verified arguments from the operand
stack into the callee frame, when the tail call reuses the current frame and the
argument shape matches. In self-recursive tail loops this runs every iteration.

**Risk.** This is the least like a peephole and the most like a calling-convention
change. Frame replacement, ownership transfer of reference arguments, and the
verifier's tail-call result-signature check all constrain it. The correctness
surface is large relative to the others, so I would only pursue it after the
in-place tail call is proven stable and only with a workload that is dominated by
self-recursion.

## How the predecoder would carry a fusion

The predecoded IR in `src/nanovm/vm_dispatch.c` already stores, per instruction,
the successor index (`next_index`), resolved branch targets (`branch_target`),
and resolved call targets (`call_target`). A fusion pass would run after
`vm_dispatch_build_function`, scan for a maintained sequence over adjacent
instructions whose only entry point is the sequence head (no branch target lands
in the middle), and rewrite the head slot to a private fused handler while
marking the covered slots unreachable. The boundary map (`offset_to_index`) makes
the "no interior branch target" check exact: a slot that any branch resolves to
cannot be absorbed. No portable opcode, assembler mnemonic, or verifier rule
changes; the verifier still validates the original instructions before the
predecoder ever runs.

## The measurement I require before accepting any of these

I do not accept a fusion because a sequence is common or because one timing
looked smaller. The acceptance gate is the existing NanoISA optimization policy
(`docs/NANOISA_OPTIMIZATION_POLICY.md`), applied per candidate:

- **Frequency floor.** The fused sequence must account for at least 1% of retired
  baseline instructions in a maintained workload, measured from the
  `make benchmark-nanoisa` instruction counters, and it must survive a second
  measurement after other lowering waste is removed.
- **Statistical improvement.** Median improvement must exceed run-to-run noise on
  the same machine, compiler, flags, workloads, and sample count, and it must
  improve or preserve both the NanoLang and Forth workload groups.
- **No hidden cost.** Retired instructions, code size, allocated bytes, retained
  values, and FFI traffic must not move the cost elsewhere.
- **Semantic equivalence.** Every relevant unit, integration, example, and
  semantic-equivalence test must still pass; the fused handler must be
  behaviorally identical to the sequence it replaces.
- **Simplicity tiebreak.** When results are statistically tied, the unfused
  dispatcher wins.

## Initial ranking

This is a ranking of where I would spend measurement effort, not a decision:

1. **Local increment** — densest and most repetitive; largest expected win;
   smallest correctness surface. Measure first.
2. **Compare-branch** — fires on every loop guard; cheap to detect in the
   predecoder; one turn plus one stack slot per guard.
3. **Local-field load** — common in aggregate-heavy code, but the getter
   heterogeneity and ownership rules raise the cost of a correct handler.
4. **Union-tag branch** — much of the naive win is already captured by
   `MATCH_TAG`; only a decoded multi-arm jump table is left, and only for wide
   matches.
5. **Tail-call fusion** — largest correctness surface and closest to a
   calling-convention change; pursue last and only for self-recursion-dominated
   workloads.

I accept a fusion only when a maintained NanoLang or Forth workload justifies it
under the policy above. Until a candidate clears that bar with repeated
measurement, it stays a candidate.
