# I repair the stale schema dispatch marker

Hosted component run39544892945, continuation job119110288530, restored its exact immutable container/checkpoint and completed raw Stage2, both-generation byte identity, native translation/hello checks and final continuation validation. Raw Stage2 ran08:45:08–10:48:33UTC; the complete generation/verification step passed at11:01:09UTC.

The subsequent unit step failed because `tests/test_nanoisa_schema.py` searched `VM_DISPATCH_BEGIN(instr.opcode)` while the actual VM dispatch uses `instr->opcode`. I changed only that test marker. The complete33-method schema suite passed in1.209seconds using the existing qualification Python environment. Initial system Python lacked PyYAML; that prerequisite failure is retained separately.

I do not claim the sanitizer unit/negative suite passed: units stopped at this schema test and negatives were skipped. This was not an ASan/UBSan diagnostic. The exact-current hosted sequence remains required. No flags, checkpoint identity, schema coverage assertions or VM implementation changed.
