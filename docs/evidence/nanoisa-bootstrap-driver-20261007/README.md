# My raw-module bootstrap driver

I replace Make's source-to-C native generations with a seed `.nvm`, two VM
self-host generations, mandatory raw equality, and separately translated native
compilers. I record source, tool, host and artifact hashes; each generation
requires the exact seed host closure. My existing native-code-generation guard
runs around the VM generations. Every command retains a terminal and duration.

My actual Make control target passes 16 methods covering source invalidation,
missing module/receipt/native artifacts, raw comparison, retained native-byte
independence, host/source/tool mutation, generated-code refusal markers,
incremental tool linking, and installation failure propagation. Three adjacent
Make toolchain methods pass. Two separate native-guard methods pass. These are
control and boundary tests, not execution of the full bootstrap.

I retain two fixture failures: the initial message fixture tries to copy an
absent `tests/__init__.py`; the dependency fixture incorrectly expects an
already-built component stage to depend on the new bootstrap script. I remove
the nonexistent copy and scope that assertion to the bootstrap consumers.
I also correct the incremental-tool fixture to supply Linux's already-built
assembler helper and model a mutable common/UTF8 object without suppressing
its mtime. I keep the real Make recipes under test.

My full clean Make bootstrap and post-cutover compiler-product gates remain
pending. I have not completed the 5.1 release or removed every historical
product dependency on the legacy emitter.
