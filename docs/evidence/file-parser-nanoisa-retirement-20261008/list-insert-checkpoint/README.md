# My list insertion checkpoint

I lower generic list insertion with staged operands, signed bounds assertions,
append-once and descending tail shifts. I preserve existing receiver aliases.
My component driver compiles the actual self-hosted emitter with C-seed shadows.

I corrected an initial compiler-source identifier mistake (`array` is reserved).
My first test invocation also started before the driver build completed; I retain
its missing-file errors without counting them as product failures. The completed
driver build exits zero. The subsequent four-method component run has one failing
subcase: C-seed record insertion runs in VM but native translation refuses
`record to optional` storage conversion. The self-hosted record case passes
VM and sanitized native execution. Integer insertion, operand order/once,
three invalid bounds through both producers and four invalid operand forms pass.

I retain complete logs, source/assembly/modules and tool hashes. The original
paired token/schema corpus remains required and unchanged. It also calls remove
and pop, which need self-hosted/native parity before this work can close.
Fresh bootstrap and full paired qualification remain open under #978.
