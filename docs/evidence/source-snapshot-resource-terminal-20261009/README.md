# Snapshot diagnostic resource terminal

I keep #980 open after its corrected 128-method corpus failed. I prepared a
nine-case diagnostic selecting exactly its seven failing and two error cases.
I retained original helper assertions and parent deadlines, captured each
subprocess command/time/output, and retained temporary fixture directories.
This diagnostic does not replace complete source-snapshot acceptance.

The first setup check referenced an absent Linux capture helper in the Darwin
checkout. I preserve that setup failure and record the helper as absent; the
Darwin Clang path does not use it. The executed diagnostic then completed 46
recorded subprocesses. Its first selected case completed; the second reached
its final reuse build before evidence writing failed with ENOSPC. Later cases
also failed to create fixture directories. The process ended with observed
exit 120 while reporting those failures, so no complete nine-case result exists.
The first 46 command records have no subprocess exception and a maximum elapsed
time of 7.939 seconds. I do not infer a cause for the earlier deadline failures.

`terminal-recovered-state.json` records a later source/probe check matching the
initial hashes and clean fdca4ee9b checkout. The host reported 1.7 GiB available
at the first disk check, then 53 GiB without any deletion by this worker.
The original failed log and partial command 47 remain here. I restarted the
same diagnostic in a new output directory only after requiring 8 GiB free;
that new run is separate and remains pending at this checkpoint.
