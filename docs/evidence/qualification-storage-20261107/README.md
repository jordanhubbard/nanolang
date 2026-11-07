# My qualification storage controls

I retain the original Xcode ENOSPC under `../xcode-reconstruction-disk-20261107/`.
The runner failed while reconstructing; its logs and final artifact were unavailable.
Unbounded successful temporary roots are present in my harnesses. I have not
established which allocation exhausted that runner.

I isolate each contract-effects phase in a fresh temporary root. My opt-in
`RECONSTRUCTION_UNITTEST` runner executes the original explicit module list in
order, one complete module per process. I keep native flags, assertions, test
methods and Make prerequisites. A failed module does not skip later modules in
that invocation. The default Make runner remains `python3 -m unittest`.

After success, I archive the entire root, including native programs and debug
sidecars. I compare every archived file's SHA-256, mode and contents, retain link
targets and directory modes, then check the original inventory again. Only then
do I remove that freshly allocated root. Failed or unverified roots remain and
have a separate artifact upload. I never discover old roots and infer ownership.
I do not remove SDKs, shared caches or another job's outputs.

I record actual device/free-space measurements every 30 seconds during each
wrapped command. Before and after phases I measure bounded directory sizes for
my workspace, temporary root and ordinary package/module caches. SDK/tool paths
receive volume measurements without a recursive size walk. I stop a command if
sampling observes less than two GiB free on its workspace/temp/output volume;
this sampling guard is not a hard allocation quota. Archives need additional
space and a failed archive preserves its original root.

My [local controls](local-controls.log) pass 25 storage and merge-policy tests.
They execute real child processes and filesystem operations, including failure
retention, bytes/modes/links/debug-sidecar archival, mutation/refusal, nested
archive lifetime and low-space refusal. My frozen Make-command fixture preserves
the original module selection. These controls do not qualify native reconstruction.

My branch-scoped recovery workflow calls the actual release-qualification
workflow with only the Xcode contract-effects matrix entry. Ordinary PR, main and
manual qualification retain both platforms and all four corpora. I require fresh
bootstrap, the complete original contract-effects/reconstruction corpus and final
provider/source checks before closing #1079.
