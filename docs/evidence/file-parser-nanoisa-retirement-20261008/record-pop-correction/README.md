# I preserve optional popped records

I copy the popped record into traced snapshot storage before shortening its
array. Boxed record values retain their children as GC roots. Empty record
pop returns void; extracting a field from that absence traps in both routes.
Checked field extraction and typed record return constrain the payload while
preserving optional source storage.

All eight paired list methods pass through C-seed and self-hosted emission,
VM execution and strict sanitized native execution. All seven raw-pop methods
pass with LLVM ASan/UBSan and leak detection: exact scalar/float values,
closures, empty record results, absent-field traps, retained record children
across owner reuse and collection, and record return/argument transport.
The typed-return control initially fails optional/record shape constraints;
I retain that failure beside the correction.

My full `make -j2 test-nvm2c CC=/opt/homebrew/opt/llvm/bin/clang` exits zero:
2,431 execution, 3,092 shape and 379 callable checks pass, together with the
seven raw-pop methods and existing Python controls. These component results
do not replace fresh installed compiler and complete paired parser gates,
Linux parity or final 5.1 qualification. Issue #978 remains open.
