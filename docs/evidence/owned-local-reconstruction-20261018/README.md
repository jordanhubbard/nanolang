# I reconstruct local resource transfers

I implement the bounded [contract](../../NANOISA_OWNED_LOCAL_RECONSTRUCTION.md)
under #999. I require actual executable verifier admission, then preserve
numeric resource layout identity, scalar fields and local ownership declarations
through a single ordered region tree into C and Nano source. I do not treat
advisory facts as authority or lower through a VM, goto or bytecode dispatcher.

## Actual producer boundary

My first probes failed because a standalone block is not accepted Nano syntax,
and an `if true` branch does not erase ownership joins. Repeated consuming
patterns are supported. Actual producer output performs ephemeral shared field
reads and consumes the hidden owner at the final unpack. I therefore admit
only the exact single-read region sequence, not arbitrary references.

My initial qualification failed an incorrectly wide test decode of packed
layout-flag bytes, and scalar neighbors lacked retained compiler executables.
I preserve that full terminal in `initial-qualification.log`; neither failure
was counted as successful qualification. GCC separately found misleading
indentation in the new helper; I corrected it and passed strict syntax checking.

## Qualification

- I passed the four focused methods with the C seed, retained Stage1 and
  retained Stage2 as actual source producers. I compare original NanoVM,
  reconstructed C at O0/O2 with ASan/UBSan, reconstructed NanoVM and its native
  ASan/UBSan execution at O0/O2. Each returns 74 after two same-shaped but distinct
  nominal resources, counter updates, moves and repeated consuming patterns.
- I check regenerated ownership version/count and packed COMPLETE|RESOURCE
  flags, ordered scalar layouts, and nominal construction indices 0 and 1.
- Nine mutated authority/nominal/move/region cases preserve both previous
  source outputs. A valid executable branch outside this slice is separately
  refused. I preserve CRC validity so these do not merely test checksum errors.
- All 14 then-current owned/scalar methods passed in 64.108 seconds. The final
  four owned methods pass in 5.292 seconds after the initializer admission guard,
  using all three source producers and both O0/O2 native modes.
- I also refuse a checksum-valid initializer-named resource entry atomically.
  My first attempt to assemble that case established that the current executable
  verifier already rejects it; I retain `initializer-probe.log` rather than
  claiming it is a valid admitted initializer. The facts guard independently
  excludes initializer naming, consistent with my existing scalar contract.

`providers.sha256` identifies copied aggregate tools. Stage1/Stage2 came from
aggregate source `28e50b802`, production closure `57652`, after its successful
`run-m17ygl7e` bootstrap. I compiled only the changed facts front end against
retained aggregate objects; I did not run a competing full bootstrap.

This is macOS focused evidence. Integrated Linux/Darwin, wider owned calls,
borrows, arrays, unions, generic identities, control flow and cross-module
reconstruction remain unfinished under rows 9538/10409 and #999.
