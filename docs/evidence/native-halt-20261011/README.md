# Ordinary native HALT

I select my cleanup-preserving unwind runtime for modules containing HALT,
including modules without handler instructions. I name this selection
`has_unwind` because both global termination and lexical effect returns use it.
I retain ordinary C returns through each caller so roots and temporary storage
are released before main returns the VM-compatible halt result. Modules without
HALT or effects keep their existing emission path.

I pass eleven boundary methods (twenty fixtures), including new handler-free
direct calls, indirect calls, suspended integer stack state and allocated string
cleanup. Generated executables run with ASan/UBSan and leak detection. My initial
suspended-stack fixture had an extra operand at its unreachable RET; I retain the
verifier rejection and correct the fixture without changing the HALT assertion.

I pass all 66 source cases per producer through the C frontend, previous fresh
seed and audited regenerated compiler, including eighteen VM/native execution
programs each. My first paired run overlaps a Make relink of nano_virt; the final
run passes with all six tool hashes unchanged. Generated native programs use
ASan/UBSan with leak detection. The translator itself is not instrumented in
this checkpoint. My broader native gate passes 2,443 core checks, 464 callable
constraints, 3,113 shape constraints and adjacent suites.
My current translator reproduces byte-identical C from the audited compiler
module. This does not establish the exact release-platform gates.

I separately reproduce mixed scalar argument refusal: `mixed-carrier.nasm`
verifies and exits zero in NanoVM, but native translation refuses the same
operation receiving integer and Boolean arguments. I leave that requirement
open under #976.

I published the worker claim through Git coordination at `2564eaaff`.
GitHub issue reads succeeded intermittently, but comment attempts failed to
connect; I do not claim issue tracking updates were posted.
