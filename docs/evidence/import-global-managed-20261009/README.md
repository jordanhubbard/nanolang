# My managed imported-global checkpoint

The baseline passes qualified/selective mutable string arrays and selective string maps in both components. C-seed qualified maps lose declared key/value metadata; record globals from both producers execute in NanoVM but fail native aggregate global storage.

I prepare two isolated C patches without changing the source inputs of the live bootstrap at 915b85284. The map patch resolves qualified imported TypeInfo from the original declaration and preserves local receiver precedence. The separate seed binary passes all seventeen existing import methods plus wrong-key, wrong-value and local-receiver refusal controls. The translator patch stores a managed record snapshot in the existing tagged global representation, using its existing collector root traversal.

Together the prototypes pass all twelve managed probes through VM and strict C11 ASan/UBSan/LSan native execution. A raw record fixture additionally retains an older owned string and nested array across 4,096 global replacements, clears the global, and requires at least two collections before successful native exit. VM output matches. Its initial fixture declares tuple for a struct-producing AGG_PACK; I retain that failed VM check and correct only the declared result type. The first prototype build script also mishandles a multiline recorded linker command; I retain that failure before joining its continuation lines.

These patches remain prepared, not integrated. The broader native suite is still running, as is bootstrap Stage2. Permanent tests, integrated qualification, nominal/ownership edge cases and complete cross-host acceptance remain required under #986.

The initial broad native suite finishes with 2,434 passes and one legacy expectation failure: test_projected_global_stores requires native record-global rejection. I retain its terminal. I prepare a test change that replaces this obsolete rejection with record-field execution checks in both function declaration orders, and rerun the full suite against the same prototype. The rerun remains live.

The corrected native suite finishes with 2,438 passes and no failures. Extra probes pass same-named record types from different modules and captured callable globals in both components. Callee-before-argument mutation passes in the C seed but self-hosted local-binding inference reports the callable type instead of the invocation result. I retain that failure in extra-baseline.log and track it under #986.

The isolated checker patch now shares callable-value argument/result validation across ordinary and qualified calls. Its full-source shadow build passes, and all six extended producer/case combinations pass, including explicit local result typing and callee snapshot before a mutating argument. The existing seventeen-method suite also passes in 12.638 seconds through the corrected temporary self-hosted component. prepared-call-result.patch remains unintegrated while bootstrap Stage2 is live.

I prepare the permanent expanded import suite, record-global retention/refusal tests and Make targets. All twenty-seven import methods pass through both isolated compiler candidates. Both record-global methods pass, including scalar overwrite followed by invalid record projection in VM and native execution. The implementation and tests remain prepared until the live baseline bootstrap terminates.

I subsequently integrate the prepared fixes and permanent tests after the baseline bootstrap passes. Integrated results and source hashes are recorded in docs/evidence/import-global-managed-20261009/integrated. A fresh bootstrap for the changed source remains required.
