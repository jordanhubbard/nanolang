# My Linux ARM64 bootstrap qualification

My pinned-generation guard correction passes all 17 bootstrap steps, byte-identical raw Stage1/Stage2 modules, exact host closure and both native smoke tests with GCC in python:3.12.12-bookworm. The installed Stage2 compiler also runs without bin/nanoc_c. The full SQLite test target then passes all seven methods in 10.024 seconds, including both self-hosted VM/native producers and strict sanitizer controls.

The isolated source copy includes the 6cc098a0c implementation changes and its own retained cache. Its manifest pins actual inputs and tools; this is not an x64 run or a claim about subsequent shadow-diagnostic source changes. I retain per-step logs, receipts and artifact hashes. Per-command SQLite temporary files were inside the removed container; I retain the complete captured test output and do not claim those files survived.
