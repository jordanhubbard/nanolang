# My canonical borrowed-source setup

My source-borrow test harness builds an emitter and shadow runner with each compiler stage before running its assertions. At frozen product `63c26ecd2723f8c480eaba4b25ebdf797f4764bd`, setup records an error under the ordinary 180-second command deadline. I retain the full focused log and will identify the exact command from its completed traceback before assigning a cause.

I measure the same current source compilation separately under a bounded 900-second observation, retain compiler hashes and elapsed time, and verify its resulting ordinary output before setting a setup-only deadline. The earlier scalar-match setup task3d463 used the same distinction; it does not by itself qualify this new source pin. I do not change ordinary producer, shadow or execution deadlines, skip assertions, or label an unexplained timeout as infrastructure.

My correction must pass every existing source-borrow method using the canonical product compilers. I keep the candidate checkout frozen and load the corrected harness from this separate branch against its source/tools. This tests the harness correction with exact product identities; it does not turn the original failed gate into a pass. The product parent remains held until a new complete acceptance is recorded.
