#ifndef NVM2C_EFFECTS_H
#define NVM2C_EFFECTS_H

/* I retain dynamic operation identities separately from lexical arm entries.
 * An arm has an empty operand stack and shares only its owner's earlier locals. */
static int effect_parameter(const Nvm2cBuf *b, uint32_t fn, uint16_t slot) {
    for (size_t i = 0; i < b->handler_count; ++i) {
        const Nvm2cHandler *h = &b->handlers[i];
        if (h->function == fn && slot >= h->start && slot - h->start < h->count) return 1;
    }
    return 0;
}

static void effect_destroy(Nvm2cBuf *b) {
    for (size_t i = 0; i < b->handler_count; ++i) free(b->handlers[i].flow);
    free(b->handlers);
    for (uint32_t i = 0; i < b->effect_result_count; ++i) free(b->effect_results[i].fields);
    free(b->effect_results);
}

static int effect_collect(Nvm2cBuf *b, const NvmModule *mod) {
    if (!b->has_effects) return 1;
    b->effect_results = calloc(mod->string_count ? mod->string_count : 1, sizeof *b->effect_results);
    if (!b->effect_results) goto memory;
    b->effect_result_count = mod->string_count;
    for (uint32_t i = 0; i < mod->string_count; ++i) {
        b->effect_results[i].kind = NVM2C_VK_UNK;
        b->effect_results[i].fields = malloc(b->record_width);
        if (!b->effect_results[i].fields) goto memory;
        memset(b->effect_results[i].fields, NVM2C_VK_UNK, b->record_width);
    }
    if (!effect_collect_entries(mod, &b->handlers, &b->handler_count)) {
        nvm2c_fail(b, "I require valid native effect control flow"); return 0;
    }
    return 1;
memory:
    nvm2c_fail(b, "I cannot allocate native effect facts"); return 0;
}

static int effect_resume_result(Nvm2cBuf *b, uint32_t fn, size_t pc,
        Nvm2cFacts *facts, Nvm2cSimSlot result) {
    for (size_t h = 0; h < b->handler_count; ++h) {
        Nvm2cHandler *handler = &b->handlers[h];
        if (handler->function != fn || !handler->flow[pc]) continue;
        Nvm2cEffectResult *out = &b->effect_results[handler->operation];
        if (!merge_fact(b, facts, &out->kind, result.kind) ||
            !merge_record_results(b, facts, out->fields, result.rec_k) ||
            !shape_equal(b, shape_variable(b, &out->shape), result.shape)) return 0;
        uint16_t tags = out->tags | result.scalar_tags;
        if (tags != out->tags) { out->tags = tags; facts->changed = 1; }
    }
    return 1;
}

static int effect_classify_perform(Nvm2cBuf *b, const NvmModule *mod, uint32_t fn,
        const DecodedInstruction *in, uint8_t *locals, Nvm2cFacts *facts,
        Nvm2cSimSlot *stack, int *depth) {
    uint32_t name = in->operands[0].u32;
    uint16_t count = in->operands[1].u16;
    if (name >= mod->string_count || count > *depth) {
        nvm2c_fail(b, "I require valid performed operation arguments"); return 0;
    }
    name = effect_operation(mod, name);
    int base = *depth - count;
    for (size_t i = 0; i < b->handler_count; ++i) {
        Nvm2cHandler *h = &b->handlers[i];
        if (h->operation != name || h->count != count) continue;
        for (uint16_t p = 0; p < count; ++p) {
            Nvm2cSimSlot v = stack[base + p];
            size_t at = (size_t)h->function * b->local_width + h->start + p;
            if (!merge_parameter(b, facts, &facts->parameters[at], v.kind) ||
                !merge_record_results(b, facts, facts->fields + at * b->record_width, v.rec_k) ||
                !shape_equal(b, shape_variable(b, &b->shape_locals[at]), v.shape)) return 0;
            if (v.kind == NVM2C_VK_UNK)
                mark_origin(locals, mod->functions[fn].local_count, v.origin, facts->parameters[at]);
        }
    }
    *depth = base;
    Nvm2cEffectResult *out = &b->effect_results[name];
    Nvm2cSimSlot result = {0};
    result.kind = out->kind; result.origin = -1; result.scalar_tags = out->tags;
    result.rec_k = sim_fields(b, out->fields, NVM2C_VK_UNK);
    if (!result.rec_k || !sim_push_slot(b, fn, stack, depth, result)) return 0;
    return shape_equal(b, stack[*depth - 1].shape, shape_variable(b, &out->shape));
}
#endif
