from pathlib import Path
root=Path('/private/tmp/nanolang-match-guards-20261009');out=Path('/private/tmp/nanolang-introspection-c-callables-20261009')
s=(root/'src/nanovirt/codegen.c').read_text().replace('} FnEntry;','    bool module_intrinsic;\n} FnEntry;',1)
a=s.index('        /* These declarations lower to module facts, not external host calls. */')
b=s.index('        return;',a)
s=s[:a]+'''        if (cg->had_error || fn_find(cg, name) >= 0) return;
        if (cg->fn_count >= MAX_FUNCTIONS) {
            cg_error(cg, 0, "I cannot register another module introspection function");
            return;
        }
        NvmFunctionEntry function = {0};
        function.name_idx = nvm_add_string(cg->module, name, (uint32_t)strlen(name));
        function.arity = intrinsic_arity;
        function.result_tag = intrinsic_result;
        function.result_count = 1;
        uint32_t index = nvm_add_function(cg->module, &function);
        if (!nvm_set_function_param_types(cg->module, index, param_tags, intrinsic_arity)) {
            cg_error(cg, 0, "I could not retain module introspection parameter types");
            return;
        }
        FnEntry *entry = &cg->functions[cg->fn_count++];
        entry->name = (char *)name;
        entry->fn_idx = index;
        entry->module_intrinsic = true;
'''+s[b:]
marker='    /* ── Pass 1.5: Compile top-level let bindings as globals ─────── */'
insert='''    /* I give metadata declarations ordinary bodies for direct and indirect calls. */
    for (int i = 0; i < cg.fn_count && !cg.had_error; ++i) {
        FnEntry *function = &cg.functions[i];
        if (!function->module_intrinsic) continue;
        uint32_t index = function->fn_idx;
        cg.code_size = 0;
        cg.local_count = 0;
        cg.local_binding_count = 0;
        cg.names_enabled = false;
        cg.current_fn_idx = index;
        cg.param_count = cg.module->functions[index].arity;
        if (cg.param_count) {
            uint16_t slot = local_add(&cg, "", 0);
            emit_op(&cg, OP_LOAD_LOCAL, (int)slot);
        }
        if (!compile_module_introspection(&cg, function->name)) {
            cg_error(&cg, 0, "I require a known module introspection operation");
            break;
        }
        emit_op(&cg, OP_RET);
        if (cg.had_error) break;
        uint32_t offset = nvm_append_code(cg.module, cg.code, cg.code_size);
        cg.module->functions[index].code_offset = offset;
        cg.module->functions[index].code_length = cg.code_size;
        cg.module->functions[index].local_count = cg.local_count;
    }

'''
assert marker in s;s=s.replace(marker,insert+marker,1);(out/'codegen.c').write_text(s)
