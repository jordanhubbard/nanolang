# My hosted verifier diagnostics

I downloaded both actual failure artifacts from CI run35590650305 after PR942 merged. I retain all1048 files losslessly in the indexed archive. Both x64 and ARM64 compiler logs identify the same two failures: test_u8_basic rejects the declared byte value type at shadow line8; token_value_bytes cannot resolve list_LexerToken_insert at shadow line21. These are existing computed-byte and generic-list tasks, not evidence that either product gate passes.

My successful uploads establish the hosted log-retention acceptance. My complete workflow and 5.1 release remain unfinished.
