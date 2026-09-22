# My SDK fork-exit probe lifecycle boundary

MAC: `task_30d1c83d335e4b618ce71b8f761c6f9a`.

My first8c93 Darwin sanitizer terminal stops in the unchanged callback probe: the child fork path allocates4096 bytes in14 libSystem `_notify_fork_child` map/set allocations; normal exit reports these and the parent rejects its exit status. I retain the complete report and do not claim that original sanitized path passed. Linux ordinary/sanitized and Darwin ordinary original probe paths pass separately.

The original child exit matters: it invokes inherited production `sdk_cache_shutdown`, which must leave the parent's callback and private directory alone. Merely changing exit to `_exit` would remove this check. My approved fixture-only adaptation captures the exact production atexit hook while forwarding its real registration unchanged. An explicit child-lifecycle mode invokes that captured hook, then checks the callback count and private directory are unchanged before `_exit`. The parent's normal exit still verifies exact callback-before-directory-removal order and exactly-once cleanup.

Only the probe's opt-in included-source build uses the narrow registration observer; ordinary default `callback` retains the original fork-to-exit path, and existing installed-SDK probe builds remain unchanged. The new explicit mode has a distinct argument and output marker. Selected sanitizer qualification must name it explicitly rather than report the original path green. I add no sanitizer suppression, disable no instrumentation, and do not alter production SDK ownership or cleanup code. The child mode checks this specific inherited lifecycle; its `_exit` does not establish general child-process leak freedom.

Source review precedes corrected execution. I will independently copy/hash-verify retained8c93 Darwin sanitizer products, rebuild only fixture-owning probes, run the corrected lifecycle mode and previously unreached loader modes under original bounds, and preserve all original results. Full threaded COP safety remains the separate exec-worker obligation.
