#ifndef NVM2C_EFFECT_EMIT_H
#define NVM2C_EFFECT_EMIT_H

static void effect_emit_runtime(Nvm2cBuf *b) {
    nvm2c_puts(b,
        "typedef struct { unsigned kind; union { int64_t t; double f; const char *s; narr_t a; nsarr_t sa; nrec_t r; nrarr_t ra; nmap_t m; nmap_value v; } data; } nfx_value;\n"
        "typedef struct nfx_frame { struct nfx_frame *lexical; void **locals; void *result; nrec_t *environment; unsigned function; int64_t fallback; } nfx_frame;\n"
        "typedef struct nfx_handler { struct nfx_handler *previous; nfx_frame *owner; unsigned operation, entry, start, count; } nfx_handler;\n"
        "typedef struct { nfx_frame *owner; unsigned entry, start, count; nfx_value *arguments; nfx_value result; } nfx_activation;\n"
        "static nfx_handler *nfx_head, *nfx_floor;\nstatic nfx_frame *nfx_unwind;\nstatic nfx_activation *nfx_enter;\nstatic unsigned nfx_depth, nfx_handlers;\nstatic int nfx_halted;\nstatic int64_t nfx_halt_result, nfx_top;\n"
        "static void nfx_dispatch(nfx_activation *activation);\n"
        "static inline void nfx_push(nfx_frame *owner, unsigned entry, unsigned operation, unsigned start, unsigned count) {\n"
        "    if (nfx_handlers == 1024) NVM2C_ABORT();\n"
        "    nfx_handler *h = malloc(sizeof *h); if (!h) NVM2C_ABORT();\n"
        "    *h = (nfx_handler){nfx_head, owner, operation, entry, start, count}; nfx_head = h; ++nfx_handlers;\n}\n"
        "static inline void nfx_pop(nfx_frame *owner, unsigned count) {\n"
        "    while (count--) { nfx_handler *h = nfx_head; if (!h || h->owner != owner) NVM2C_ABORT(); nfx_head = h->previous; free(h); --nfx_handlers; }\n}\n"
        "static inline void nfx_prune(nfx_handler *saved) {\n"
        "    while (nfx_head != saved) { nfx_handler *h = nfx_head; if (!h) NVM2C_ABORT(); nfx_head = h->previous; free(h); --nfx_handlers; }\n}\n"
        "static inline nfx_activation *nfx_perform(unsigned operation, unsigned count, nfx_value *arguments) {\n"
        "    nfx_handler *h = nfx_head; while (h && h != nfx_floor && h->operation != operation) h = h->previous;\n"
        "    if (!h || h == nfx_floor || h->count != count) NVM2C_ABORT();\n"
        "    nfx_activation *activation = calloc(1, sizeof *activation); if (!activation) NVM2C_ABORT();\n"
        "    activation->owner = h->owner; activation->entry = h->entry; activation->start = h->start; activation->count = count; activation->arguments = arguments;\n"
        "    nfx_dispatch(activation); return activation;\n}\n");
}

static void effect_emit_dispatch(Nvm2cBuf *b, const NvmModule *mod, const uint8_t *kinds) {
    nvm2c_puts(b, "static void nfx_dispatch(nfx_activation *activation) {\n    if (nfx_enter) NVM2C_ABORT();\n    nfx_enter = activation;\n    switch (activation->owner->function) {\n");
    for (uint32_t f = 0; f < mod->function_count; ++f) {
        int has_handler = 0;
        for (size_t h = 0; h < b->handler_count; ++h) if (b->handlers[h].function == f) has_handler = 1;
        if (!has_handler || !b->emitted_functions[f]) continue;
        const NvmFunctionEntry *fn = &mod->functions[f];
        char name[64]; fn_c_name(mod, f, name, sizeof name);
        int record = fn->result_count == 1 && aggregate_value_tag(fn->result_tag);
        nvm2c_printf(b, "    case %u: {\n", f);
        if (record) nvm2c_puts(b, "    nrec_t *result = calloc(1, sizeof *result); if (!result) NVM2C_ABORT();\n");
        nvm2c_printf(b, "    (void)%s(%s", name, record ? "result" : "");
        if (fn->upvalue_count) nvm2c_printf(b, "%sactivation->owner->environment", record ? ", " : "");
        for (uint16_t i = 0; i < fn->arity; ++i) {
            uint8_t k = fn_local_kind(b, kinds, f, i);
            nvm2c_printf(b, "%s%s(%s){0}", i || record || fn->upvalue_count ? ", " : "", k == NVM2C_VK_REC ? "&" : "", c_local_type(k));
        }
        nvm2c_puts(b, ");\n");
        if (record) nvm2c_puts(b, "    free(result);\n");
        nvm2c_puts(b, "    break; }\n");
    }
    nvm2c_puts(b, "    default: NVM2C_ABORT();\n    }\n    if (nfx_enter) NVM2C_ABORT();\n}\n");
}

static void effect_emit_frame(Nvm2cBuf *b, const NvmModule *mod, uint32_t idx, const uint8_t *kinds) {
    const NvmFunctionEntry *fn = &mod->functions[idx];
    nvm2c_printf(b,
        "    (void)nfx_push; (void)nfx_pop; (void)nfx_perform;\n"
        "    if (++nfx_depth > 1024) NVM2C_ABORT();\n"
        "    nfx_activation *nactivation = nfx_enter; nfx_enter = NULL;\n"
        "    int nresumed = 0;\n    nfx_handler *nfx_saved = nfx_head;\n"
        "    void *nfx_locals[%u] = {0};\n"
        "    nfx_frame nfx = {nactivation ? nactivation->owner : NULL, nfx_locals, NULL, %s, %u, nfx_top};\n",
        fn->local_count ? fn->local_count : 1, fn->upvalue_count ? "nenv" : "NULL", idx);
    for (uint16_t i = 0; i < fn->local_count; ++i) {
        uint8_t kind = fn_local_kind(b, kinds, idx, i);
        char local[32]; physical_local_operand(local, b, kinds, idx, i);
        nvm2c_printf(b, "    %s *p%u = &%s;\n", c_local_type(kind), i, local);
        nvm2c_printf(b, "    if (nactivation && %u < nactivation->start) p%u = nactivation->owner->locals[%u];\n    nfx_locals[%u] = p%u;\n", i, i, i, i, i);
        nvm2c_printf(b, "    if (nactivation && %u >= nactivation->start && %u - nactivation->start < nactivation->count) {\n", i, i);
        nvm2c_printf(b, "    if (nactivation->arguments[%u - nactivation->start].kind != %u) NVM2C_ABORT();\n", i, kind);
        nvm2c_printf(b, "    *p%u = nactivation->arguments[%u - nactivation->start].data.%s;\n    }\n", i, i, stack_array_name(kind));
    }
}

static void effect_emit_return(Nvm2cBuf *b, const NvmFunctionEntry *fn, const char *rt, int record) {
    nvm2c_puts(b, "    if (nactivation && !nresumed && !nfx_unwind && !nfx_halted) {\n    nfx_frame *owner = nfx.lexical; while (owner->lexical) owner = owner->lexical;\n");
    if (record) nvm2c_puts(b, "    *(nrec_t *)owner->result = *nresult;\n");
    else if (fn->result_count) nvm2c_printf(b, "    *(%s *)owner->result = nresult;\n", rt);
    nvm2c_puts(b, "    nfx_unwind = owner;\n    }\n    if (nfx_unwind == &nfx) nfx_unwind = NULL;\n    nfx_prune(nfx_saved); --nfx_depth;\n");
}

static int effect_emit_perform(Nvm2cBuf *b, const NvmModule *mod, Nvm2cStack *st, const DecodedInstruction *in) {
    uint16_t count = in->operands[1].u16;
    uint32_t op = effect_operation(mod, in->operands[0].u32);
    if (count > st->sp) { nvm2c_fail(b, "I require native perform arguments"); return 0; }
    nvm2c_printf(b, "    {\n    nfx_value *arguments = calloc(%u, sizeof *arguments); if (!arguments) NVM2C_ABORT();\n", count ? count : 1);
    for (uint16_t i = count; i > 0; --i) {
        uint8_t kind;
        int slot = stack_pop_kind(b, st, &kind);
        if (b->failed) return 0;
        nvm2c_printf(b, "    arguments[%u].kind = %u; arguments[%u].data.%s = %s[%d];\n", i - 1, kind, i - 1, stack_array_name(kind), stack_array_name(kind), slot);
    }
    nvm2c_printf(b, "    nfx_activation *activation = nfx_perform(%u, %u, arguments);\n    free(arguments);\n    if (nfx_unwind || nfx_halted) { free(activation); goto L_return; }\n", op, count);
    uint8_t kind = b->effect_results[op].kind;
    if (kind == NVM2C_VK_UNK) kind = NVM2C_VK_VALUE;
    nvm2c_printf(b, "    if (activation->result.kind != %u) NVM2C_ABORT();\n", kind);
    char value[64]; snprintf(value, sizeof value, "activation->result.data.%s", stack_array_name(kind));
    int slot = -1;
    if (kind == NVM2C_VK_STR) slot = stack_push_str(b, st, value);
    else if (kind == NVM2C_VK_FLOAT) slot = stack_push_float(b, st, value);
    else if (kind == NVM2C_VK_BOOL) slot = stack_push_bool(b, st, value);
    else if (word_array_storage(kind)) slot = stack_push_iarray(b, st, value, kind);
    else if (kind == NVM2C_VK_SARR) slot = stack_push_sarr(b, st, value);
    else if (kind == NVM2C_VK_MAP) slot = stack_push_map(b, st, value);
    else if (kind == NVM2C_VK_FUNC) slot = stack_push_function(b, st, value);
    else if (kind == NVM2C_VK_VALUE) slot = stack_push_value(b, st, value);
    else if (kind == NVM2C_VK_REC || kind == NVM2C_VK_RARR) {
        slot = kind == NVM2C_VK_REC ? stack_push_rec(b, st, value) : stack_push_rarr(b, st, value);
        if (slot >= 0) memcpy(kind == NVM2C_VK_REC ? st->rec_k[slot] : st->rarr_k[slot], b->effect_results[op].fields, b->record_width);
    } else slot = stack_push_temp(b, st, value);
    nvm2c_puts(b, "    free(activation);\n    }\n");
    return slot >= 0 && !b->failed;
}
#endif
