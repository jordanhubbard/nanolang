# Passive Parallelism Contract

**Status:** Normative 5.0 design; not the current implementation

I make independent computation visible without making concurrency part of
program meaning. In 5.0, `par` and `flow` are effect-checked declarations that
produce an ordinary dependency graph in NanoISA metadata. Their reference
execution is serial and deterministic. A VM or translator may ignore the
eligibility metadata and still produce the same result.

This contract does not promise parallel execution or speedup. Threads are not
semantics.

## Current Boundary

Neither the C frontend nor the self-hosted frontend currently implements the
complete syntax, effect analysis, dependency analysis, or conformance matrix
below. NanoISA does not yet carry the required eligibility metadata, and its
verifier does not yet validate such metadata. I therefore make no
production-readiness claim for passive parallelism in the current release.

Everything after this section is the 5.0 target. An example is a specification,
not evidence that either compiler accepts it today.

## The 5.0 Slice

5.0 includes exactly these pieces:

- `par` blocks whose bindings are mutually independent and effect-safe.
- `flow` blocks whose dependencies form an acyclic dataflow graph.
- A closed effect summary sufficient to reject observable and unsafe work.
- NanoISA metadata that records nodes, dependencies, source order, effects,
  and resource accesses.
- Verifier checks that make the metadata safe to trust as optimization advice.
- Deterministic serial reference semantics shared by NanoVM and every AOT
  translator.

The following remain outside 5.0:

- A runtime scheduler, worker pool, work stealing, or parallel NanoVM dispatch.
- Async I/O, futures, `await`, or overlap between I/O and computation.
- Automatic parallel `map`, `filter`, or `fold`, including associativity claims.
- `soa struct`, automatic SoA conversion, SIMD layout promises, and pipeline
  fusion.
- Hardware-specific scheduling, benchmark thresholds, or speedup claims.
- Shared mutation coordinated by locks, atomics, or implicit synchronization.

Those are separate features. Eligibility metadata permits an optimization; it
does not smuggle the optimization into this contract.

## Canonical Syntax

`par` contains independent `let` bindings:

```nano
par {
    let left: int = (score_left input)
    let right: int = (score_right input)
}
let total: int = (+ left right)
```

No initializer in a `par` block may refer to a binding declared by that block.
The bindings enter the enclosing scope together after all initializers finish.
Their names are not visible inside any initializer, including the initializer
that declares the name.

`flow` contains bindings whose uses declare their dependencies:

```nano
flow {
    let left: int = (score_left input)
    let right: int = (score_right input)
    let total: int = (+ left right)
    let result: int = (normalize total)
}
```

A `flow` binding may refer to another binding in the same block regardless of
textual order. The resulting dependency graph must be acyclic. All bindings
enter the enclosing scope after the block finishes.

Only `let` bindings are permitted directly inside either block in 5.0. `mut`,
assignment, loops, conditionals, `return`, `break`, `continue`, nested `par` or
`flow`, declarations, and expression statements are rejected. This small
surface gives both frontends one graph to lower and one verifier rule to apply.

## Serial Reference Semantics

The meaning of an accepted block is a deterministic serial evaluation:

1. Build the dependency graph described below.
2. Repeatedly select the ready node with the lowest source-order index.
3. Evaluate that initializer to completion.
4. Make its result available to dependent nodes.
5. Commit all block bindings to the enclosing scope after every node succeeds.

For `par`, every node has an empty internal dependency set, so this rule is
ordinary source-order evaluation. For `flow`, it is a stable topological order.
The frontend emits that order in metadata; the verifier recomputes it rather
than trusting the claim.

An initializer trap stops the block at that node under serial execution. No
later binding becomes visible. Eligibility is restricted enough that an
alternative execution order cannot add an observable effect before that trap.
This rule does not promise recovery from traps or speculative execution.

NanoVM may always execute the serial order. An AOT translator may always emit
that same order. If an implementation evaluates eligible nodes concurrently,
it must preserve the serial result and trap behavior. The same valid `.nvm`
must therefore remain correct when no parallel facility exists.

## Effects

Each initializer receives a closed effect summary. A node is eligible only
when its transitive call graph is known and contains none of these effects:

- host, file, network, terminal, clock, random, process, thread, or FFI I/O;
- writes to globals, captured mutable storage, heap objects, or arguments;
- reads from mutable globals, captured mutable storage, volatile state, clock,
  random state, or host state;
- allocation or finalization with user-visible callbacks;
- calls whose effect summary is absent, open, imported without a verified
  summary, or marked unsafe;
- traps whose occurrence depends on another node's mutation or external state.

Allocation of ordinary private GC values is permitted when allocation identity,
address, timing, and collection are unobservable to the program. Reading
immutable inputs and constructing immutable results are permitted.

`@pure` is a checked summary, not an assertion that overrides analysis. The C
and self-hosted frontends must derive the same closed summary. A declaration
whose body contradicts `@pure` is rejected whether or not it appears in a
parallelism block.

## Dependencies

A node depends on every in-block binding read by its initializer. Dependency
edges point from producer to consumer. Repeated reads produce one edge. Reads
through a pure helper count transitively when the helper captures an in-block
binding.

For `par`, any internal edge is an error. For `flow`, every internal read must
have a matching edge, every edge must name a node in the same block, and the
graph must be acyclic. Unused nodes are permitted because purity makes their
evaluation unobservable except for their value or a deterministic trap.

The analysis is lexical and place-based. Aliases do not erase a dependency.
If two reads originate from the same immutable input they do not conflict. If
the frontend cannot prove what a read names, it rejects the block rather than
marking it eligible.

## Resources

An initializer may borrow an immutable ordinary value. It may not create,
move, consume, return, capture, or borrow a resource-bearing value as defined
by the affine ownership contract. It may not call a function with a
resource-bearing parameter or result, even through a shared borrow.

This deliberately rejects disjoint resources. Proving that two handles refer
to independent external state requires an effect and capability model beyond
5.0. A file handle is still I/O when two files happen to be different.

Mutable places are also rejected, even when two nodes appear to use different
fields or array indices. Field- and index-disjoint mutation needs a finer place
model and is not part of this contract.

## NanoISA Eligibility Metadata

Each `par` or `flow` block lowers to ordinary serial NanoISA instructions plus
one metadata record. The instructions are authoritative program semantics; the
record is validated optimization advice.

The record contains:

| Field | Meaning |
|---|---|
| `kind` | `par` or `flow` |
| `function` | Owning NanoISA function |
| `entry` / `exit` | Instruction range for the complete serial block |
| `nodes` | One entry per source binding |
| `node.id` | Dense block-local identifier |
| `node.order` | Dense source-order index |
| `node.entry` / `node.exit` | Contiguous instruction range for one initializer |
| `node.result` | Local receiving the completed value |
| `node.dependencies` | Sorted producer node identifiers |
| `node.effects` | Closed, empty observable-effect set |
| `node.reads` | Canonical immutable places read outside the block |
| `node.resources` | Empty resource access set |

The metadata does not contain thread counts, scheduling policy, costs, target
features, or expected speedup. Such facts do not affect validity and do not
belong in a portable `.nvm`.

Both frontends must emit byte-for-byte equivalent records after normal module
canonicalization for the same source construct. Differences in private local
numbering are normalized before comparison.

## Verifier Rules

The NanoISA verifier rejects an eligibility record unless all of these hold:

- The record and every node range lie within one function and on instruction
  boundaries.
- Node identifiers and source-order indices are unique, dense, and complete.
- Node ranges are nonempty, disjoint, ordered by `node.order`, and exactly
  partition the block range apart from declared binding-commit instructions.
- Every dependency names a node in the same record, has no duplicate, and the
  graph is acyclic.
- A `par` record has no dependency edges.
- Every read of an in-block result has a declared producer edge, and every
  declared edge corresponds to such a read.
- Control flow cannot enter or leave the middle of a node, cross between node
  bodies, or bypass the block commit.
- Each node writes only its declared result and private temporaries.
- Effect metadata agrees with verified calls and opcodes; an unknown, unsafe,
  mutable, resource, or I/O operation rejects the record.
- The recomputed stable topological order agrees with the serialized code.

A malformed record rejects the module. Silently discarding it would let one
backend accept a contract another backend found unsafe. A module without these
records remains an ordinary serial module and makes no passive-parallelism
claim.

## Rejection Examples

An internal dependency is not `par`:

```nano
par {
    let first: int = (prepare input)
    let second: int = (finish first)
}
```

The same graph is valid `flow` only when both calls are effect-safe:

```nano
flow {
    let first: int = (prepare input)
    let second: int = (finish first)
}
```

I/O is never eligible in 5.0:

```nano
par {
    let left: string = (read_file left_path)
    let right: string = (read_file right_path)
}
```

Neither is mutation disguised as independence:

```nano
par {
    let left: int = (increment_counter counter)
    let right: int = (increment_counter counter)
}
```

## Conformance Matrix

Every row requires a positive and negative test in the C frontend, the
self-hosted frontend, the NanoISA verifier, serial NanoVM, and each maintained
AOT translator where execution applies. Passing one frontend is not
conformance.

| Rule | Positive case | Negative case |
|---|---|---|
| `par` independence | two pure nodes reading one immutable input | one node reads another node |
| `flow` dependencies | pure diamond graph | missing or extra dependency edge |
| Acyclic graph | stable topological chain | direct or indirect cycle |
| Lexical scope | results visible after block | in-block name escapes its valid scope |
| Pure call | closed transitively pure helper | false `@pure` declaration |
| Unknown call | verified closed call graph | import or indirect call without summary |
| Immutable read | shared immutable input | mutable global or captured mutable read |
| Writes | private result construction | global, captured, argument, field, or index write |
| I/O | pure arithmetic | file, network, terminal, clock, random, process, or FFI use |
| Resources | ordinary GC value | create, move, consume, return, or borrow resource |
| Syntax | direct `let` bindings | `mut`, control flow, statement, or nested block |
| Node ranges | exact disjoint partition | overlap, gap, cross-node branch, or mid-node entry |
| Stable order | source order among ready nodes | serialized order disagrees with metadata |
| Trap behavior | deterministic pure trap | externally dependent or effectful trap |
| Backend agreement | one `.nvm` matches serial VM and AOT result | backend requires scheduler metadata to be correct |
| Metadata portability | policy-free canonical record | thread count, target feature, or speedup claim |

Frontend tests must compare canonical metadata, not only accept or reject the
source. Verifier tests must also mutate each structural field in a known-good
module and demonstrate rejection. Execution tests run the same `.nvm` through
serial NanoVM and every maintained AOT translator and compare result, output,
exit status, and trap location.

## Release Claim

I may call this contract implemented only when every matrix row has named tests,
the C and self-hosted frontends agree, the verifier rejects unsafe metadata,
and the same `.nvm` passes serial NanoVM and AOT translation without a parallel
scheduler. Until then, documentation must label `par`, `flow`, and their
metadata as 5.0 target behavior.

Scheduler optimization, async I/O, SoA layout, and hardware speedups require
separate accepted tasks with their own measurements. This contract supplies no
evidence for those claims.
