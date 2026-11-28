# My rebuilt Stage2 bounded stacks

I sample my genuine Stage2 image `8c2c8660e05fb97a4bf0017c0de2a9b7f3a7b5fc114128583b62f5cb01dd7fa4` at exact source `cfd62a78d5f8afaa6bbd7cc835d75ca97e3caf00`. I bind my published original cold bootstrap (`cdcdd3956f2ea632899d964d85383be261a1ea42`) and separate phase observation (`9b1c871c1b62b60444711aee4cdb52d769dd79d9`), rather than substituting a development image or a hand-written bootstrap receipt.

I use one new admitted private root, my unchanged emitter fixture and existing MI debugger driver. My independent debugger control and actual adopted-child cleanup control pass before execution. I request at most eight seconds each after typecheck-start and program-lowering-start, within a 300-second outer bound and ordinary ten-second shadow policy. I terminate my inferior at 230.855894 seconds. My debugger exits `0`; owned cleanup leaves no inferior or adopted child.

## My typecheck-start window

I retain 8 stacks over 8.000157 seconds. My observed frame-presence counts are:

- `nl___nano_module___25_typecheck_advisory_core`: 8
- `nl___nano_module___32_nanoisa_check_and_emit_native`: 8
- `nl_compile_program_inputs`: 8
- `main`: 8
- `nl_main`: 8
- `nl___nano_module___25_check_function`: 7
- `__memcpy_sve`: 5
- `nl___nano_module___25_check_expr_node`: 5
- `nl___nano_module___25_check_owned_statement_list`: 5
- `nl___nano_module___25_check_statement_list`: 3
- `nl___nano_module___25_check_block`: 3
- `nl___nano_module___25_check_set_statement`: 3
- `nl___nano_module___25_env_copy_symbols`: 2
- `nl___nano_module___25_check_record_callable_fields`: 2
- `nl___nano_module___25_check_return_statement`: 2
- `nl___nano_module___25_merge_struct_metadata`: 2
- `memcpy@plt`: 1
- `nl___nano_module___9_parser_get_struct_def`: 1
- `nl___nano_module___25_tc_layout_visit`: 1
- `nl___nano_module___25_tc_validate_layouts`: 1
- `nroot_slot`: 1
- `nroot_add`: 1
- `__memset_generic`: 1
- `nrarr_get`: 1
- `nl___nano_module___25_env_lookup`: 1

## My program-lowering-start window

I retain 8 stacks over 8.000151 seconds. My observed frame-presence counts are:

- `nl___nano_module___32_nanoisa_check_and_emit_native`: 5
- `nl_compile_program_inputs`: 5
- `nl___nano_module___32_nisa_emit_module_context`: 5
- `main`: 5
- `nl_main`: 5
- `nl___nano_module___32_nisa_emit_block`: 5
- `nl___nano_module___32_nisa_emit_stmt`: 5
- `nl___nano_module___32_nisa_emit_statements`: 5
- `nrarr_get`: 4
- `__memcpy_sve`: 4
- `nl___nano_module___32_nisa_fn_index`: 4
- `nl___nano_module___32_nisa_emit_if`: 4
- `nl___nano_module___32_nisa_field_shape_supported_seen`: 4
- `nl___nano_module___32_nisa_emit_function_effects`: 4
- `nl___nano_module___32_nisa_record_shape_supported_seen`: 4
- `nl___nano_module___32_nisa_call_index`: 3
- `nl___nano_module___32_nisa_is_pin_record`: 3
- `nl___nano_module___32_nisa_emit_expected_expr`: 3
- `nl___nano_module___32_nisa_emit_expr`: 3
- `nl___nano_module___32_nisa_emit_call`: 2
- `nl___nano_module___32_nisa_result_tag`: 2
- `nl___nano_module___32_nisa_expr_type`: 2
- `nl___nano_module___32_nisa_type_ok`: 1
- `nl___nano_module___32_nisa_type_ok_seen`: 1
- `nl___nano_module___32_nisa_emit_typed_let`: 1

All 3,971 exact source inputs and producer/image/tool/host/debugger/driver/observer guards match. I retain complete MI and stderr streams, phase markers, waits and products. My collector refuses a real nonterminal attempt before creating this archive.

These are bounded observations with debugger and new-root/cache effects. Frame counts are not cost fractions, a sole cause or a performance comparison. My original emitter12 target still fails in setup at its original300 limit; these samples establish no product repair, native consumer acceptance, paired qualification, full pure runtime emission, cutover or release acceptance.

Run `python3 read_archive.py` to verify all 4,743 records and their raw/compressed bytes, exact Git inputs, published parents, controls, window bounds, complete transcript, cleanup and guards. My manifest SHA-256 is `ea0ff0ae2ce494e9ae6f58bec4799aa48c89833a644a596a9dd6089266b6bc70`. I preserve previous roots and failures.
