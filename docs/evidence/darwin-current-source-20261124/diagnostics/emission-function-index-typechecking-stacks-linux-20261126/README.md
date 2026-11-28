# My rebuilt Stage2 bounded stacks

I sample my genuine Stage2 image `7f431c60c2f344f877f031be669ac64d8b17bd198f218c4fe0d15754795a6695` at exact source `8fa1025477ca2b7db9266f1019b7192de20175d1`. I bind my published original cold bootstrap (`0581ef3a1448b91f88a6d7334ce9ff7a343f4699`) and separate phase observation (`2224c2cafdf94b5c18d39b5ceb42ed6f296e33d9`), rather than substituting a development image or a hand-written bootstrap receipt.

I use one new admitted private root, my unchanged emitter fixture and existing MI debugger driver. My independent debugger control and actual adopted-child cleanup control pass before execution. I request one window of at most eight seconds after typecheck-start, within a 300-second outer bound and ordinary ten-second shadow policy. I terminate my inferior at 106.388296 seconds. My debugger exits `0`; owned cleanup leaves no inferior or adopted child.

## My typecheck-start window

I retain 8 stacks over 8.000166 seconds. My observed frame-presence counts are:

- `main`: 7
- `nl_compile_program_inputs`: 7
- `nl___nano_module___25_typecheck_advisory_core`: 7
- `nl_main`: 7
- `nl___nano_module___32_nanoisa_check_and_emit_native`: 7
- `nl___nano_module___25_check_owned_statement_list`: 7
- `__memcpy_sve`: 6
- `nl___nano_module___25_check_function`: 6
- `nl___nano_module___25_check_expr_node`: 6
- `nrarr_get`: 5
- `nl___nano_module___25_env_lookup`: 5
- `nl___nano_module___25_check_block`: 4
- `nl___nano_module___25_check_statement_list`: 4
- `nl___nano_module___25_tc_bound_name`: 2
- `nl___nano_module___25_tc_validate_layouts`: 1
- `nl___nano_module___25_tc_layout_visit`: 1
- `__memset_generic`: 1
- `nl___nano_module___25_check_return_statement`: 1
- `nroot_slot`: 1
- `nmap_collect`: 1
- `nroot_contains`: 1
- `nroot_add`: 1
- `nl___nano_module___25_check_let_statement_named`: 1
- `nl___nano_module___25_check_let_statement`: 1

All 3,971 exact source inputs and producer/image/tool/host/debugger/driver/observer guards match. I retain complete MI and stderr streams, phase markers, waits and products. My collector refuses a real nonterminal attempt before creating this archive.

These are bounded observations with debugger and new-root/cache effects. Frame counts are not cost fractions, a sole cause or a performance comparison. My original emitter12 target still fails in setup at its original300 limit; these samples establish no product repair, native consumer acceptance, paired qualification, full pure runtime emission, cutover or release acceptance.

Run `python3 read_archive.py` to verify all 4,747 records and their raw/compressed bytes, exact Git inputs, published parents, controls, window bounds, complete transcript, cleanup and guards. My manifest SHA-256 is `d9cee4b6ae2db4c113f42d4140d054a195b5451e0b17410eb22acc3826e325c3`. I preserve previous roots and failures.
