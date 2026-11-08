# My hosted Darwin compiler-contracts checkpoint

I pass the complete compiler-contracts Xcode 27 leg of
[run 42397278317](https://github.com/jordanhubbard/nanolang/actions/runs/42397278317/job/128477253106).
My [exact job and steps](job.json) show fresh compiler generations, the complete
selected native compiler/managed-runtime corpus and final source/provider checks
passing. I retain full compressed bootstrap, corpus and final provider logs.

My requested head is `435dae34cc2ce74d5210806d26ef1ebb9f9e4b7a`; my actual
PR merge checkout is `a6ba0d210667fcc65d91778fcfcccaa8b2cfb917`. I compare the
artifact's complete source-tree record byte-for-byte with both Git trees; both
resolve to `6f0a4a71268ce3e22f611363d7befa83461b260c`. My [summary](summary.json)
retains artifact size/hash, CRC result, original unittest batch counts/timings,
reported skips and nine unchanged providers. My eight-method preliminary batch
reports one skip; its nonverbose output does not identify that method. I retain
that boundary rather than claiming zero skips. My [complete member index](artifact-members.json)
records original bytes, hashes and modes; the full original ZIP remains under
`~/.worktrees/nanolang/audits/2026-11-07/hosted-435-qualification/compiler-contracts-xcode.zip`.

My general compiler version file reports Homebrew clang 23.1.4. My independently
captured selected wrappers invoke Apple clang 21.0.0 from Xcode 27.2 and pin its
SDK; I retain both records and the actual wrapper contents. This capture does
not establish verifier-owned host-toolchain/SDK admission.

I retain the original private union/record test boundaries, including explicit
no-public-enrollment results. I do not infer complete ordinary source authority
or native lifecycle acceptance from those private checks. Other Darwin corpus
legs, the contract-effects disk recovery, complete JUnit, full integration,
GitGuardian #1076 and native lifecycle graph/receipt requirements remain open.
