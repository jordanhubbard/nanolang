# My mixed ordinary/resource fixed point

At frozen compiler revision af2b25da6 I pass all 17 raw NanoISA bootstrap steps, including exact immutable host closure, byte-identical Stage1/Stage2 modules, native linking/smoke execution and installed operation without the C seed. Source, tool and pinned host-library hashes remain unchanged. I retain the complete manifest, installed receipt, host-cache logs and terminal.

I then pass all nine module-identity methods through the default four producers (legacy C, C-seed NanoISA/VM/native, Stage1 and Stage2), all 28 import methods through both installed stages, and five linker-flag configurations through each installed stage. I retain exact method timings in verification.json and gate.log.

This validates the integrated mixed ordinary/resource changes locally on Darwin. It does not include the later indexed module-binding candidate or prove Linux/full 5.1 release acceptance. Concurrent isolated profiling created other cache generations, but my recorded closure and raw stage equality still pass.
