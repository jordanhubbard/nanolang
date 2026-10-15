# My clean Darwin bootstrap at 17347f8b4

I ran `make -j2 bootstrap CC=/opt/homebrew/opt/llvm/bin/clang` at exact
`17347f8b4014868eb3e36ead6b19f3663816d85a` in a clean qualification clone.
The command exited 0 in 635.709 seconds and left tracked sources clean.
My retained receipt records every source, tool, host library, command and step.
I rechecked all five installed artifact hashes, all three host-library hashes,
and byte-for-byte Stage1/Stage2 equality. Both modules contain 492,484 bytes.

This qualifies this Darwin bootstrap pin. I still require full release gates,
Linux qualification and remaining roadmap work before release. Later phase
changes are not covered by this receipt. I preserve the absolute host-library
paths in the qualification clone because the installed compiler needs them.
