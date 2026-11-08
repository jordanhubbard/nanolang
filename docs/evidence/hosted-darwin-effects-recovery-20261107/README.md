# My completed Darwin storage recovery

My original Xcode runner ENOSPC remains preserved in [the incident record](../xcode-reconstruction-disk-20261107/README.md). I do not establish its historical allocation cause from unavailable logs.

Run [42411895451](https://github.com/jordanhubbard/nanolang/actions/runs/42411895451), job 128521863128, succeeds at exact source f0b99facb4b2415761c980f21e69d644326b3825. Fresh bootstrap, all 91 metadata/effects methods, both compiler-product derivations, and all 334 reconstruction methods pass, with zero reported skips. All nine provider checks pass. My complete hosted source tree matches `git ls-tree -r` of that source; I do not equate this commit with the earlier integration head.

Capture uses the selected Apple clang 21.0.0 with Xcode/SDK 27.2 and recorded compiler/assembler/wrapper hashes. The separate general build compiler remains identified in compiler.txt. I retain the complete original phase logs, bootstrap manifests, source tree, provider hashes and worker volume journals. My minimum sampled free space is 25533837312 bytes; sampled measurements do not prove a hard allocation quota.

I independently verify all 62 complete phase archives inside the fully downloaded 347,414,539-byte artifact 13594819667: ZIP CRC, each compressed archive SHA256, every one of 35,822 archived entries' kind/mode and file bytes/hash, and matching complete inventories. Every wrapped command ends with status zero. Journals record successful archive verification before removal of that phase's task-owned root. I stream this verification without extracting native binaries or SDK content locally. [verification.json](verification.json) records the exact archive identities and phase labels; complete compressed archives remain at the private path it names.

This completes the bounded retention and full Xcode recovery acceptance under #1079. Full current integration, the separate Darwin source-snapshot failures under #980/#1049, and native lifecycle/release gates remain open. I do not close them from this checkpoint.
