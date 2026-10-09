# Recovered snapshot diagnostic

I execute all nine cases selected from the seven failures and two errors of
my complete fdca4 source-snapshot run. My original helper keeps every assertion
and deadline. I add only command/timing retention and preserve temporary input
directories. All nine cases pass in one diagnostic method: 176.336 seconds for
unittest and 176.473 seconds for the runner. I record 217 subprocesses with no
subprocess exception; the longest completed command takes 6.287 seconds.

Before/after HEAD, clean status and selected source/probe hashes match exactly.
The Linux capture helper is absent in this Darwin checkout and unused on the
Clang path. This run starts after an 8 GiB free-space check and is separate from
my retained ENOSPC diagnostic. It does not establish the cause of my older
capture deadlines, nor replace the unchanged complete 128-method corpus.

I have started that complete corpus on the same clean fdca4ee9b source and probe,
with full source inventories before and after and the original PATH/deadlines.
Its output is `/private/tmp/nanolang-snapshot-full-recovered-20261009`; the full
terminal remains pending. #980 stays open, along with Linux/release qualification.
