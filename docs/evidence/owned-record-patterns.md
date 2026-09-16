# My owned-record pattern checkpoint

I lower `let Record { field, ... } = value` into one private temporary and
field bindings. The temporary name cannot be spelled by source identifiers.
Only parser-created nodes carry whole-owner and projection markers. I require
the initializer's nominal record type and every field exactly once before
ownership checking. I resolve the aggregate and give resource-bearing fields
their own obligations; this does not free ordinary GC backing storage.

Both parsers retain these markers in their ASTs. My self-hosted AST schema
and generated C layout agree. My flow passes reject ordinary partial resource
field moves while allowing the complete pattern's generated projections.

The runtime cases also exposed four existing implementation gaps:

- My self-hosted emitter treated unsafe-block bindings as globals and emitted
  unsafe blocks without a C scope. I now classify these bindings as local and
  preserve the outer generation environment at block exit.
- My VM retained local names after block exit. I retire the names without
  reclaiming their slots and bound generated symbol metadata to its scope.
- My C seed inferred an array binding's element type only from literals. I
  reuse expression element inference for fields, aliases and declared calls.
- My C reflection generator emitted an orphan `else` for empty records. I
  emit unconditional fallback returns after the field dispatch.

## Tests

On Darwin, `make -j1 bootstrap3 nano_virt nano_vm` passes the bootstrap and
installed-compiler independence checks. This is not canonical fixed-point
equality.

`python3 -m unittest tests.test_owned_record_patterns` passes 12 methods on my
C seed, Stage 1, Stage 2 and NanoVM. Positive cases compile and execute shadows
and the resulting program. Negative cases preserve the prior output artifact;
ownership rejections require an ownership diagnostic. Cases cover complete,
empty and nested patterns; invalid fields and nominal types; initializer
evaluation once; unsafe shadowing; ordinary array fields and aliases;
use-after-move; unresolved transferred fields; and partial moves. The three
existing simulated-resource fixtures also compile and execute on each path.
Their helpers dismantle simulated handles, not operating-system files.

`make -j1 test-nanovirt` passes 74 code-generation tests, including captures.
Parser, typechecker, transpiler and evaluator unit targets pass, followed by
all 21 `test-one-ir-compiler` methods. Generated schema and changed-file shadow
policy checks pass. The broader quick gate is a separate release requirement.
The final rebuilt combined run passes 43 methods across frontend parity,
ownership boundaries, lexical scope and owned-record patterns.

## Boundary

This checkpoint does not implement borrows, resource captures or match-payload
ownership. It does not establish generic substitution, module-owned nominal
identity, allocation-failure conformance or ownership verification in NanoISA.
The complete ownership task and release gate remain open.

MAC: `task_826838b808f340968c526f849276b913`,
`task_44c3da4d710e4a939c8879b9b5ecd15e`,
`task_414223eec172441b88619f5e9744c5c1`,
`task_4f76b90bc54d47b5b4de531930869431`,
`task_94d61370acb44862a22b716ad0aa5ef8`.
