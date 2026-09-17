# Text write error reporting

I share `nl_write_file_text` across interpreter writes/appends, the VM writer,
module writes/appends, and generated-native writes/appends. I return 0 only
after a full `fwrite` and successful `fclose`. I return -1 for null input,
failed open, short write or failed close, and always close an opened stream.
Write and append modes remain distinct. Failure can leave partial output;
this is not atomic replacement or durable-storage confirmation.

## Verification

`make test-file-write` rebuilds the C seed and VM tools and runs a fault probe
for all seven production wrapper bodies. The probe extracts those bodies
from current source (and literal generated-C output), compiles them with the
real shared helper, and injects stdio failures. It checks exact open/write/close
counts, write-versus-append mode, open failure, short write, close failure and
empty success. This isolates the production bodies; it is not failure
injection into complete interpreter or VM processes.

`make test-nvm2c` passes 771 checks, including its existing generated-code
writer fault probes and real filesystem artifact write/append/read/copy tests.
`tests/test_file_text.py` passes its C-seed/VM/module compilation and execution
regression in 11.059 seconds. `git diff --check` passes.

MAC `task_46cb33d875724130a6f767d998f3014b` tracks this fix. My claim was refused
with `agent_status_unavailable`; I record evidence without forcing ownership.
Full compiler AOT acceptance and the release remain unfinished.
