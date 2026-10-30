# I qualify integrated wrapping pretests

At `62673cd2c` I pass all 14 wrapping, observation, loop and assertion-order methods in 263.150 seconds through rebuilt NanoVirt and both fresh `74a565150` stages. I retain exact provider hashes and command receipts. This Python reconstruction change does not rebuild either source compiler.

I qualify exact INT ADD/SUB/MUL endpoint wrapping, eager condition evaluation and final false tests through original verified VM modules, reconstructed C and Nano, regenerated verification/execution and O0/O2 ASan/UBSan native execution. Supplemental operation counters measure generated C only. DIV remains outside this pretest profile; division is total, so the existing assertion-order control does not claim a division trap. Helper/local limits and atomic refusal controls remain unchanged.

I attach the new suite to `test-owned-reconstruction`. This is Darwin qualification, not full #999 completion or hosted Linux evidence. The component receipt retains the earlier accepted-source probes and refusal evidence. Log normalization removes trailing display whitespace; original and retained hashes are recorded.
