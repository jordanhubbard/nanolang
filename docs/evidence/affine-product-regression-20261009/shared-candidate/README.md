# My tested isolated shared record candidate

I retain my unintegrated implementation patch and candidate checks for #987.
My compiler source remains frozen at 5dc3a17fb during its independent bootstrap.

My prepared shared verifier distinguishes complete copyable records from resource
records, preserves layout identity at joins/locals/calls/results, and rejects
owner construction through ordinary aggregate operations. My native translator
uses its existing reference-counted record carrier and failure cleanup.

My final self-hosted component passes all nine unchanged module-identity methods
in 5.008 seconds. My updated full-source compiler shadows pass, including a useful
ordinary receiver and nested copy/call/result shadow. Fourteen malformed raw cases
refuse without replacing prior output. Positive raw records containing integers
or owned strings pass verification, VM and native execution, and every allocation
failure through success frees all tracked roots with ASan/UBSan/LSan enabled.

My adjacent suites pass 413 affine-state and 1,150 bytecode checks; allocation
variants pass 445 and 1,787. The corrected full native suite passes 2,438 checks.
I retain the initial temporary link omission, missing test include path and
missing native CLI argument. The first verifier candidate checks accidentally
loaded the original library through embedded absolute artifact paths; their
failures do not test the corrected shared verifier. I rebuilt the library and
relinked only the candidate compiler module artifact reference before claiming
the later passes. My provenance records both compiler artifact hashes.

This candidate is not integrated, has no fresh installed-stage fixed point, and
has not passed Linux qualification. Broader ordinary/resource graphs, borrowed
calls, collections and remaining full 5.1 scope stay open. I do not close #987.
