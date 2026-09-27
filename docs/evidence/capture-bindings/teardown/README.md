# My trapped-frame destruction checkpoint

My source checkpoint bb08c4e13 detaches and releases callable owners from
surviving VM frames after successful callback shutdown. Callback-shutdown
refusal still returns before teardown mutation. I found the omitted ownership
edge by source audit and reviewed the correction before executing its control.

My new real indirect-call control traps with a managed tuple captured in an
owned closure, then checks frame/handler and heap reclamation at destruction.
The unchanged complete make test-nanovm gate passes on Linux GCC and Darwin
Apple Clang. Each reports274642 VM assertions, zero failures, plus the existing
substring, callback, heap-allocation and stack-allocation controls.

My first953-input package omitted VM fixture dependencies. Both hosts stop at
compilation of missing substring/callback fixture files; no unit fixture runs.
The corrected989-input package adds every tests/nanovm input from bb08c4e13,
without changing product source, driver, assertions or deadlines.

I preserve the first Darwin extraction refusal caused by starting extraction
before the archive transfer completed. A subsequent corrected archive extraction
used the default umask and failed the driver input-identity assertion before
compilation: eight0664 input modes became0644; all corresponding bytes matched.
A fresh extraction with tar -p preserves archive modes and passes the same
identity check. I retain previous directories and do not overwrite their runs.
The corrected archive SHA256 is
5a1b6d41ccb47102748f417fc670621df935a51da491656cbbc575cfa2d34952.

My completed reports retain exact input hashes/modes before and after execution,
compiler identity, command outputs, exit statuses, deadlines and process-group
cleanup. Darwin reports are copied locally; its successful tree is under
/Users/jkh/nanolang-qualification/vm-destroy-bb08-preserved-modes.
This directory contains reports and manifests, not a complete binary archive.

This qualifies the bounded teardown correction. Shared-capture wire admission,
initialization verification, full frame/effect integration, every backend and
my final5.1 release gate remain open.
