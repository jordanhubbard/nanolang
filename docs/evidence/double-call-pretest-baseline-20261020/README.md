# I retain the unchanged eager double-call pretest baseline

I use the exact existing OwnedCallPretestReconstruction.source(double=True). Seed and both stages compile, verify and execute it successfully. All have caller/helper locals37/3 and three caller CALL_REF instructions (one top-level witness plus two in the pretest). Facts succeeds; both reconstructed outputs refuse the current one-call pretest guard and preserve prior output. This is a baseline, not new admission or reconstructed execution.
