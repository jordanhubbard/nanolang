# My original Linux 806 route census

I run the unchanged complete original four-route audit after the [fresh raw bootstrap](../generic-array-raw-linux-20261122/README.md), at frozen806668bb32326915f5e217026eb95f83d8b35625. The controller exits1 after984.536s. Each route selects the same237 cases with zero skips: legacy C-native and C-produced NanoISA VM pass237; native Stage1 and Stage2 each pass215 and fail the same22.

I retain source/tool/host/artifact guards, all four complete XML reports and runner logs, remaining failure outputs, storage journals, verified successful temporary archives and metadata for the two empty retained failure directories. The original runner removes successful transient per-test products; I claim no unavailable product bytes. The original resource-effect compile hits its120s Perl alarm on both native stages; the logs establish the alarm, without establishing its cause. I change no assertion or deadline and perform no unchanged retry.

The22 failures include unsupported foreign/opaque source boundaries, nominal row/enum and private-shadow behavior, the HashMap contextual join regression tracked in #1186 and regex exact-builtin-ABI refusal tracked in #1187. These failures keep #1081/#976 acceptance open. This audit does not include complete C-produced-NanoISA-to-C-AOT coverage; legacy direct C-native execution does not establish that route.

I verify all83 retained records and recompute selection, pass/fail and skip counts with `python3 read_archive.py`. Its pinned base manifest supplies the separately verified raw receipt and source/image archive. My reader checks successful temporary archive contents without extracting them. The manifest preserves the final guards; the old root remains unchanged. No current-source or Darwin acceptance is implied.

I also retain four bounded localization commands: the old bd native compiler accepts the unchanged Map fixture, current d8 seed refuses it, the C producer emits the original regex module, and the actual translator refuses its exact builtin ABI. Translation refusal is not native execution. These controls mutate no frozen source and retain all original guards.
