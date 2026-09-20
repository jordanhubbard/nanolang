# My final parser provider integration

I qualify the bounded ordinary refresh at
`7011339acc4fbffeea12e72fdba1ff4d7fdd8892` against actual canonical PR929
`a2c32d186f1c6b3c2645626e5540a1272bf52866`. My
[report manifest](file-service-parser-final/report-sha256.json),
[artifact index](file-service-parser-final/artifact-index.json), and
[scope summary](file-service-parser-final/seal-summary.json) retain the measured
inputs, commands, terminals and products.

I verify all 15 incoming canonical source/fixture paths, all 40 File parser and
fixture paths against the frozen CBC qualification, and 815 unchanged direct
C/Nano/module producer inputs outside NanoISA/VM. Incoming production changes
are confined to six NanoISA/VM files: counted allocator test hooks and raw
CAST_U8 support. I preserve the independent acceptance of those changes.

I copy exactly four hash-verified CBC compiler binaries into separate fresh
trees on Linux and puck. I copy no objects or module caches. These binaries
retain their CBC bootstrap provenance, including their older linked NanoISA
providers; I do not call them current fully relinked compilers. I build the
affected NanoISA/VM/NanoVirt and common provider closure afresh, then use fresh
module and publisher products. Every phase checks the four retained compiler
hashes before and after use. No bootstrap or sanitizer matrix is repeated.

All ordinary phases pass on both hosts:

| Phase | Linux GCC seconds | puck Apple Clang seconds |
| --- | ---: | ---: |
| Fresh providers | 20.712 | 9.019 |
| Complete paired parser method and C ownership/refusal fixture | 299.685 | 312.850 |
| Actual publisher controls | 5.678 | 7.022 |
| Wrapper controls | 3.722 | 7.937 |

I preserve the complete actual parser/publisher declarations, ordinary match
controls, selected shadows, schema generator comparisons and unresolved-service
refusals. I retain explicit compiler/SDK/tool selection and the native-only
Linux GCC installation flag; Wasm selection is unchanged. These ordinary
results do not expand the original selected-provider sanitizer claim.

Both initial source maps agree on all 54,637 tracked paths. Final current checks
verify the same 54,637 sources and 12 selected tools on each host, plus 482 Linux
and 431 puck products. Final library/module products are explicitly postphase
observations; phase bin/obj maps retain their actual boundaries. I retain special
publisher files in the transport archive and record their types without following
links or materializing FIFOs during local extraction.

I observe no new unexpected terminal in this refresh. My original 5a0e and CBC
full matrices, their first failures and their exact bootstrap/sanitizer provenance
remain separate. Companion resolution, executable lowering, full generated
service behavior and general parser/union ownership acceptance remain required
open parent work. This parser/retention milestone does not admit File source
execution.
