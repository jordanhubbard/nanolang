# I qualify bounded exclusive-helper control in the aggregate

I integrate component557f7b5b2 as b2634d342 and wire its gate at50dddecf2. I rebuild nvm2hl with `make -f Makefile.gnu -o bootstrap3 -j1 nvm2hl`, then pass all12 new methods and four old-profile byte goldens in191.594s. NanoVirt plus fresh Stage1/Stage2 from combined282d7a9f1 produce the actual modules. All eight compiler/runtime hashes remain unchanged; frontend and runtime compiler sources are unchanged by this reconstruction-only component.

Both unchanged anchors qualify original VM/native and reconstructed C/Nano/native behavior. The control-flow anchor fits128/21 locals and its full-tree shadow60/21; the return anchor fits120/12 with shadow20/12. All original assertions and snapshots across calls remain. The component receipt retains the original132-local refusal before narrow adjacent snapshot folding, and the actual130-local variant still refuses ordinary publication. Helper loop proofs bound rank updates and work independently from the existing128/two-function limits. The returns source actually lowers without a backedge; I do not claim it as return-loop bytecode evidence.

I retain exact build/test logs and provider pins. Hosted Linux/Darwin and full #999 remain open, including consumed-owner reassignment representation and other finite required families.
