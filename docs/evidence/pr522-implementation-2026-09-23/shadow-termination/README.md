# My interpreted shadow termination evidence

I report the child's signal, nonzero exit status, or missing completion instead
of collapsing all three into one failure. The existing deadline and completion
channel still control publication. MAC `task_4c272f7837ec4112ac73db3325e122c0` tracks this diagnostic repair.

The full ten-method imported-shadow gate passes nine methods, including the
expanded supervision control for hangs, bounds failure, abort, exit(0) and
exit(7). Its existing source-only imported-call case fails in the C backend;
MAC `task_ca371e7dc51c4b238848404880294c92` tracks that separate acceptance defect.
I retain the complete failing terminal and do not claim a green full gate.

I rebuild every compiler C object at `-O0` with ASan/UBSan and rerun the whole
compiler with the unchanged 60-second shadow deadline. External libraries and
generated module objects remain ordinary. Leak detection is disabled as in the
hosted sanitizer worker. The supervisor reports signal 4. The retained macOS
crash extract records EXC_BAD_ACCESS/KERN_PROTECTION_FAILURE inside the stack
guard, reports SIGILL, and separately records segmentation-fault termination.
I preserve that distinction rather than rewriting the OS report. It records
435 original frames, with recursive frames elided in the displayed backtrace.
Disassembly of `eval_call_impl` shows a 0xa9a0-byte stack reservation before
additional ASan handling. MAC `task_e43e2a52806f4609bf05de6c41e263b2` tracks reducing recursive frame use.

This establishes a local stack-guard failure. It does not establish the cause
of the separate Linux 60-second timeout. Compressed logs retain exact bytes;
`logs.json` records uncompressed sizes and hashes. `crash-extract.json` retains
relevant crash fields and the hash of the complete local report without host
identifiers unrelated to this failure.

The focused imported-child control and native supervision fixture both pass
in `shadow-status-focused.log`, including deadline validation and child cleanup.
