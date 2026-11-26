# My rebuilt Stage2 bounded stacks

I sample my genuine Stage2 image `8d7c240099edbe11cde499a268705f34660c2d940da4895378fc2a2961c886a5` at exact source `771a14330475bcd784d7cb19a215d6a4f15728d4`. I bind my published original cold bootstrap (`08c223f945474f459364c24d5e8767e4ede1b0ab`) and separate phase observation (`fa182043809cc83131d1b81f26841549597f15f1`), rather than substituting a development image or a hand-written bootstrap receipt.

I use one new admitted private root, my unchanged emitter fixture and existing MI debugger driver. My independent debugger control and actual adopted-child cleanup control pass before execution. I request one window of at most eight seconds after shadow-lowering-start, within a 300-second outer bound and ordinary ten-second shadow policy. I terminate my inferior at 172.219717 seconds. My debugger exits `0`; owned cleanup leaves no inferior or adopted child.

## My shadow-lowering-start window

I retain 8 stacks over 8.000110 seconds. My observed frame-presence counts are:

- `nl_main`: 8
- `nl_compile_program_inputs`: 8
- `main`: 8
- `nl___nano_module___32_nanoisa_check_and_emit_native`: 8
- `nl___nano_module___32_nisa_emit_shadows_native_inner`: 7
- `nl___nano_module___32_nisa_emit_statements`: 7
- `nl___nano_module___32_nisa_emit_function`: 7
- `nl___nano_module___32_nisa_emit_function_effects`: 7
- `nl___nano_module___32_nisa_emit_block`: 7
- `nl___nano_module___32_nisa_emit_stmt`: 7
- `nl___nano_module___32_nisa_call_index`: 5
- `nl___nano_module___32_nisa_emit_expr`: 5
- `__memcpy_sve`: 5
- `nl___nano_module___32_nisa_expr_type`: 4
- `nl___nano_module___32_nisa_emit_assert`: 4
- `nl___nano_module___32_nisa_emit_expected_expr`: 3
- `nl___nano_module___32_nisa_emit_typed_let`: 3
- `nrarr_get`: 3
- `nl___nano_module___32_nisa_emit_call`: 2
- `nl___nano_module___32_nisa_fn_index`: 2
- `do_system`: 1
- `__GI___wait4`: 1
- `__GI___waitpid`: 1
- `nl___nano_module___26_native_phase_marker`: 1
- `nvalue_equal`: 1

All 3,971 exact source inputs and producer/image/tool/host/debugger/driver/observer guards match. I retain complete MI and stderr streams, phase markers, waits and products. My collector refuses a real nonterminal attempt before creating this archive.

These are bounded observations with debugger and new-root/cache effects. Frame counts are not cost fractions, a sole cause or a performance comparison. My original emitter12 target still fails in setup at its original300 limit; these samples establish no product repair, native consumer acceptance, paired qualification, full pure runtime emission, cutover or release acceptance.

Run `python3 read_archive.py` to verify all 5,200 records and their raw/compressed bytes, exact Git inputs, published parents, controls, window bounds, complete transcript, cleanup and guards. My manifest SHA-256 is `b7683d4db66498d9fa6a741fafcbf1d65da03189cf36a1e16f7d49060a74ef8d`. I preserve previous roots and failures.
