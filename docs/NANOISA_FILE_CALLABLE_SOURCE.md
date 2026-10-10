# I lower File helper values in both compilers

Under #989 I check and lower noncapturing named helpers independently in C and
Nano. My callable annotations retain structural parameter/result types, exact
nominal identities and shared/exclusive borrow modes. I support nested callable
parameters and results, local reassignment, returned callees, higher-order calls
and qualified imported helper values. My [source fixture](../tests/fixtures/file_indirect_source.nano)
combines these with File ownership transfer and actual borrowed writes.

I evaluate the callee before arguments, retain it in a copy slot, and release
that slot before invoking it. I emit `FUNCREF`, `CALL_INDIRECT`, or the explicit
`FILE_CALL_INDIRECT_REFS` form. My existing owner staging and borrow regions cover
arguments; my complete indirect target query checks all candidates, including
helpers selected through parameters and results. The completed call graph must
remain acyclic. An unresolved target, capture, duplicate owner, overlapping
exclusive borrow, incompatible signature or unsourced callable refuses output.
An ordinary function type carries no purity guarantee, so I refuse indirect
calls from pure functions.

I select the complete required shadows before publishing bytecode or a native
product. A failed shadow leaves an existing destination intact. Compile-time and
runtime File grants remain separate. My native products use the checked public
indirect runtime and do not embed the VM dispatcher.

I order all named helpers before all shadows, retaining dependency order and
source order within each group. Both producers use reference-map constants
first, then catalog names and function names; import signatures precede helper
signatures. My C serializer performs this ordering on its checked File product,
without modifying the source module. Allocation failures preserve output
pointers and lengths. My Nano serializer independently emits the same bytes.

`tests.test_service_drivers` exercises C seed and bytecode drivers;
`tests.test_nano_service_driver` exercises the Nano compiler in VM/native forms
and compares both callable fixtures against C-produced bytes. These source tests
complement my [public indirect runtime corpus](NANOISA_FILE_INDIRECT_PUBLIC.md).
They do not substitute for a fresh bootstrap fixed point, platform qualification,
mixed-profile acceptance or the remaining 5.1 release gates.
