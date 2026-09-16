# My MAC dependency shadows stay local

Importing `stdlib/mac.nano` used to run six installed-CLI queries while checking
its shadows: list, open, ready, priority, show and statistics. A real CLI can
spend the compiler's ten-second shadow deadline contacting its hub. I reproduced
those six invocations with a recording CLI fixture; no live hub was needed.

My public query signatures remain unchanged. They delegate to query
implementations with an explicit command runner, passing the real `exec_command`.
Their module shadows pass a deterministic local runner and check returned task
fields and statistics. I keep real command execution, stream capture and exit
status in `exec_command`; I do not add a production test mode or change the
shadow deadline.

`tests/test_mac_command_boundary.py` passes four test methods across the native
and VM backends. It checks shell argument quoting, single command execution,
stdout/stderr/status from that execution, and both exec-once corpus fixtures.
A hostile installed-CLI fixture records no calls during those corpus compiles
or executions. A separate CLI fixture exercises all six public query wrappers
and checks their exact commands and invocation counts during root shadows and
ordinary execution.

MAC `task_5e530abdfdd3459c978149ae4b25f80f` tracks this correction.
