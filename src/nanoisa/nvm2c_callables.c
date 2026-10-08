#include "nvm2c_callables.h"
#include "../nanovm/vm_decode.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    NvmShapeId callee, result;
    NvmShapeId *arguments;
    uint32_t *linked;
    size_t linked_count;
    uint16_t arity, results;
} IndirectCall;

typedef struct {
    const NvmModule *module;
    NvmCallableAnalysis *out;
    NvmShapeId **locals, *results, *globals;
    size_t global_count;
    uint32_t field_count;
    IndirectCall *calls;
    size_t call_count;
} Analysis;

typedef struct { NvmShapeId *stack; size_t count; int set; } Frame;

static int fail(Analysis *a, const char *message) {
    if (!a->out->error[0]) snprintf(a->out->error, sizeof a->out->error, "%s", message);
    return 0;
}
static NvmShapeId fresh(Analysis *a) {
    return nvm_shape_new(&a->out->shapes, NVM_SHAPE_UNKNOWN);
}
static int flow(Analysis *a, NvmShapeId source, NvmShapeId destination) {
    return nvm_shape_convert(&a->out->shapes, source, destination);
}
static NvmShapeId field(Analysis *a, NvmShapeId value, uint32_t index) {
    return nvm_shape_child(&a->out->shapes, value, index);
}
static int connect_call(Analysis *a, uint32_t target, const NvmShapeId *arguments,
                        uint16_t arity, uint16_t results, NvmShapeId result) {
    if (target >= a->module->function_count)
        return fail(a, "I require a module-local callable target");
    const NvmFunctionEntry *fn = &a->module->functions[target];
    if (fn->arity != arity || fn->result_count != results || results > 1)
        return fail(a, "I require matching callable argument and result counts");
    for (uint16_t i = 0; i < arity; ++i)
        if (!flow(a, arguments[i], a->locals[target][i])) return 0;
    return !results || flow(a, a->results[target], result);
}
static int edge(Analysis *a, Frame *frames, uint32_t destination,
                const NvmShapeId *stack, size_t count, uint32_t *work, size_t *tail) {
    Frame *frame = &frames[destination];
    if (!frame->set) {
        frame->stack = calloc(count ? count : 1, sizeof *frame->stack);
        if (!frame->stack) return fail(a, "I cannot allocate callable control-flow storage");
        frame->count = count;
        frame->set = 1;
        for (size_t i = 0; i < count; ++i) frame->stack[i] = fresh(a);
        work[(*tail)++] = destination;
    }
    if (frame->count != count) return fail(a, "I require equal callable stack heights at joins");
    for (size_t i = 0; i < count; ++i)
        if (!flow(a, stack[i], frame->stack[i])) return 0;
    return 1;
}

static int collect_function(Analysis *a, uint32_t function) {
    VmDecodedFunction decoded = {0};
    const NvmFunctionEntry *fn = &a->module->functions[function];
    char error[VM_DECODE_ERROR_SIZE];
    if (!vm_decode_function(a->module, function, &decoded, error)) return fail(a, error);
    size_t count = decoded.instruction_count;
    Frame *frames = calloc(count + 1, sizeof *frames);
    uint32_t *work = calloc(count + 1, sizeof *work);
    NvmShapeId *stack = NULL;
    int ok = 0;
    if (!frames || !work) { fail(a, "I cannot allocate callable analysis frames"); goto done; }
    size_t head = 0, tail = 0;
    if (!edge(a, frames, 0, NULL, 0, work, &tail)) goto done;
    while (head < tail) {
        uint32_t at = work[head++];
        if (at == count) continue;
        const VmDecodedInstruction *decoded_instruction = &decoded.instructions[at];
        const DecodedInstruction *in = &decoded_instruction->instruction;
        const InstructionInfo *info = isa_get_info(in->opcode);
        if (!info) { fail(a, "I require known callable-analysis instructions"); goto done; }
        int pop = info->pop_count, push = info->push_count;
        uint32_t target = in->operands[0].u32;
        switch (in->opcode) {
        case OP_CALL: case OP_TAIL_CALL:
            if (target >= a->module->function_count) { fail(a, "I require a valid direct call target"); goto done; }
            pop = a->module->functions[target].arity;
            push = a->module->functions[target].result_count; break;
        case OP_CALL_EXTERN:
            if (target >= a->module->import_count) { fail(a, "I require a valid callable-analysis import"); goto done; }
            pop = a->module->imports[target].param_count;
            push = a->module->imports[target].return_type != TAG_VOID; break;
        case OP_CALL_INDIRECT: pop = in->operands[0].u16 + 1; push = in->operands[1].u16; break;
        case OP_AGG_PACK: pop = in->operands[3].u16; push = 1; break;
        case OP_ARR_LITERAL: pop = in->operands[1].u16; push = 1; break;
        case OP_RET: pop = fn->result_count; push = 0; break;
        default: break;
        }
        if (pop < 0 || push < 0 || push > 3) {
            fail(a, "I require an explicit stack contract for callable analysis"); goto done;
        }
        size_t before = frames[at].count;
        if ((size_t)pop > before) {
            snprintf(error, sizeof error, "I found callable-analysis stack underflow in function %u at %u (%s: %zu available, %d required)",
                     function, decoded_instruction->byte_offset, info->name, before, pop);
            fail(a, error); goto done;
        }
        size_t base = before - (size_t)pop, after = base + (size_t)push;
        size_t capacity = before > after ? before : after;
        free(stack);
        stack = calloc(capacity ? capacity : 1, sizeof *stack);
        if (!stack) { fail(a, "I cannot allocate a callable-analysis stack"); goto done; }
        if (before) memcpy(stack, frames[at].stack, before * sizeof *stack);
        NvmShapeId output[3] = {0};
        for (int i = 0; i < push; ++i) output[i] = fresh(a);
        switch (in->opcode) {
        case OP_FUNCREF:
            if (target >= a->module->function_count) { fail(a, "I require a valid function reference"); goto done; }
            if (!nvm_shape_function_add(&a->out->shapes, output[0], target)) goto done;
            break;
        case OP_LOAD_LOCAL: case OP_STORE_LOCAL: {
            uint16_t slot = in->operands[0].u16;
            if (slot >= fn->local_count) { fail(a, "I require a valid callable local"); goto done; }
            if (in->opcode == OP_LOAD_LOCAL) output[0] = a->locals[function][slot];
            else if (!flow(a, stack[base], a->locals[function][slot])) goto done;
            break;
        }
        case OP_LOAD_GLOBAL: case OP_STORE_GLOBAL:
            if (target >= a->global_count) { fail(a, "I require a valid callable global"); goto done; }
            if (in->opcode == OP_LOAD_GLOBAL) output[0] = a->globals[target];
            else if (!flow(a, stack[base], a->globals[target])) goto done;
            break;
        case OP_DUP: output[0] = output[1] = stack[base]; break;
        case OP_SWAP: output[0] = stack[base + 1]; output[1] = stack[base]; break;
        case OP_ROT3: output[0] = stack[base + 2]; output[1] = stack[base]; output[2] = stack[base + 1]; break;
        case OP_AGG_PACK:
            if (!nvm_shape_unify(&a->out->shapes, output[0], nvm_shape_new(&a->out->shapes, NVM_SHAPE_RECORD))) goto done;
            for (int i = 0; i < pop; ++i)
                if (!flow(a, stack[base + (size_t)i], field(a, output[0], (uint32_t)i))) goto done;
            break;
        case OP_AGG_GET: output[0] = field(a, stack[base], in->operands[0].u16); break;
        case OP_ARR_NEW:
            if (!nvm_shape_unify(&a->out->shapes, output[0], nvm_shape_new(&a->out->shapes, NVM_SHAPE_ARRAY))) goto done;
            break;
        case OP_ARR_LITERAL:
            if (!nvm_shape_unify(&a->out->shapes, output[0], nvm_shape_new(&a->out->shapes, NVM_SHAPE_ARRAY))) goto done;
            for (int i = 0; i < pop; ++i)
                if (!flow(a, stack[base + (size_t)i], field(a, output[0], 0))) goto done;
            break;
        case OP_ARR_GET: output[0] = field(a, stack[base], 0); break;
        case OP_ARR_SET: case OP_ARR_PUSH:
            output[0] = stack[base];
            if (!flow(a, stack[before - 1], field(a, output[0], 0))) goto done;
            break;
        case OP_HM_NEW:
            if (!nvm_shape_unify(&a->out->shapes, output[0], nvm_shape_new(&a->out->shapes, NVM_SHAPE_MAP))) goto done;
            break;
        case OP_HM_GET: output[0] = field(a, stack[base], 1); break;
        case OP_HM_SET:
            output[0] = stack[base];
            if (!flow(a, stack[base + 1], field(a, output[0], 0)) ||
                !flow(a, stack[base + 2], field(a, output[0], 1))) goto done;
            break;
        case OP_HM_DELETE: output[0] = stack[base]; break;
        case OP_CALL: case OP_TAIL_CALL:
            if (!connect_call(a, target, stack + base, (uint16_t)pop, (uint16_t)push, output[0])) goto done;
            if (in->opcode == OP_TAIL_CALL && push && !flow(a, output[0], a->results[function])) goto done;
            break;
        case OP_CALL_INDIRECT: {
            if (push > 1) { fail(a, "I require at most one native callable result"); goto done; }
            if (a->call_count >= SIZE_MAX / sizeof *a->calls) {
                fail(a, "I cannot represent indirect-call constraints"); goto done;
            }
            IndirectCall *next = realloc(a->calls, (a->call_count + 1) * sizeof *next);
            if (!next) { fail(a, "I cannot allocate indirect-call constraints"); goto done; }
            a->calls = next;
            IndirectCall *call = &a->calls[a->call_count++];
            *call = (IndirectCall){.callee = stack[before - 1], .result = output[0],
                                   .arity = (uint16_t)(pop - 1), .results = (uint16_t)push};
            call->arguments = calloc(call->arity ? call->arity : 1, sizeof *call->arguments);
            if (!call->arguments) { fail(a, "I cannot retain callable arguments"); goto done; }
            memcpy(call->arguments, stack + base, call->arity * sizeof *call->arguments);
            a->out->callees[function][decoded_instruction->byte_offset] = call->callee;
            break;
        }
        case OP_RET:
            if (pop > 1 || (pop && !flow(a, stack[base], a->results[function]))) goto done;
            break;
        default: break;
        }
        for (int i = 0; i < push; ++i) stack[base + (size_t)i] = output[i];
        if (a->out->shapes.error) goto done;
        if (in->opcode == OP_RET || in->opcode == OP_HALT || in->opcode == OP_TAIL_CALL) continue;
        if (in->opcode == OP_JMP || in->opcode == OP_JMP_TRUE || in->opcode == OP_JMP_FALSE) {
            int64_t offset = (int64_t)decoded_instruction->byte_offset + in->operands[0].i32;
            if (offset < 0 || (uint64_t)offset > decoded.code_size ||
                !vm_decoded_function_has_boundary(&decoded, (uint32_t)offset)) {
                fail(a, "I require a valid callable branch target"); goto done;
            }
            uint32_t destination = offset == decoded.code_size ? (uint32_t)count : (uint32_t)(vm_decoded_function_at(&decoded, (uint32_t)offset) - decoded.instructions);
            if (!edge(a, frames, destination, stack, after, work, &tail)) goto done;
            if (in->opcode == OP_JMP) continue;
        }
        if (!edge(a, frames, at + 1, stack, after, work, &tail)) goto done;
    }
    ok = 1;
done:
    if (frames) for (size_t i = 0; i <= count; ++i) free(frames[i].stack);
    free(frames); free(work); free(stack);
    vm_decoded_function_free(&decoded);
    return ok;
}

/* I preserve mutable element identity through copied array/map handles, including arrays
 * nested in copied records. A later write through either alias reaches reads
 * through the other; scalar function assignments still flow only forward. */
static int connect_array_aliases(Analysis *a, int *changed) {
    NvmShapeGraph *g = &a->out->shapes;
    typedef struct { NvmShapeId source, target; } Pair;
    for (size_t c = 0; c < g->conversion_count; ++c) {
        size_t count = 1, cursor = 0;
        Pair *pairs = malloc(sizeof *pairs);
        if (!pairs) return fail(a, "I cannot allocate callable alias constraints");
        pairs[0] = (Pair){g->conversions[c].source, g->conversions[c].target};
        while (cursor < count && !g->error) {
            Pair pair = pairs[cursor++];
            pair.source = nvm_shape_root(g, pair.source);
            pair.target = nvm_shape_root(g, pair.target);
            if (pair.source == pair.target) continue;
            NvmShapeKind kind = nvm_shape_kind(g, pair.source);
            if (kind != nvm_shape_kind(g, pair.target)) continue;
            if (kind == NVM_SHAPE_ARRAY || kind == NVM_SHAPE_MAP) {
                uint32_t fields = kind == NVM_SHAPE_MAP ? 2 : 1;
                for (uint32_t f = 0; f < fields && !g->error; ++f) {
                    NvmShapeId from = nvm_shape_child(g, pair.source, f);
                    NvmShapeId to = nvm_shape_child(g, pair.target, f);
                    if (nvm_shape_root(g, from) != nvm_shape_root(g, to)) {
                        if (!nvm_shape_unify(g, from, to)) break;
                        *changed = 1;
                    }
                }
            } else if (kind == NVM_SHAPE_RECORD) {
                for (uint32_t field_index = 0; field_index < a->field_count; ++field_index) {
                    NvmShapeId from = nvm_shape_lookup(g, pair.source, field_index);
                    NvmShapeId to = nvm_shape_lookup(g, pair.target, field_index);
                    if (!from || !to || from == to) continue;
                    size_t seen = 0;
                    while (seen < count &&
                           (nvm_shape_root(g, pairs[seen].source) != from ||
                            nvm_shape_root(g, pairs[seen].target) != to)) ++seen;
                    if (seen < count) continue;
                    if (count >= SIZE_MAX / sizeof *pairs) {
                        fail(a, "I cannot represent callable alias constraints"); free(pairs); return 0;
                    }
                    Pair *next = realloc(pairs, (count + 1) * sizeof *pairs);
                    if (!next) { free(pairs); return fail(a, "I cannot grow callable alias constraints"); }
                    pairs = next; pairs[count++] = (Pair){from, to};
                }
            }
        }
        free(pairs);
        if (g->error) return 0;
    }
    return 1;
}

void nvm_callable_destroy(NvmCallableAnalysis *out) {
    if (out->callees) for (uint32_t i = 0; i < out->function_count; ++i) free(out->callees[i]);
    free(out->callees); free(out->code_lengths);
    nvm_shape_destroy(&out->shapes);
    memset(out, 0, sizeof *out);
}

NvmShapeId nvm_callable_at(NvmCallableAnalysis *out, uint32_t function, uint32_t offset) {
    if (out->error[0] || function >= out->function_count || offset >= out->code_lengths[function]) return 0;
    NvmShapeId shape = out->callees[function][offset];
    if (!shape || nvm_shape_kind(&out->shapes, shape) != NVM_SHAPE_FUNCTION) return 0;
    return shape;
}

int nvm_callable_analyze(const NvmModule *module, NvmCallableAnalysis *out) {
    memset(out, 0, sizeof *out);
    Analysis a = {.module = module, .out = out};
    int ok = 0;
    if (!module || !module->functions || !module->function_count) return fail(&a, "I require callable module functions");
    out->function_count = module->function_count;
    out->callees = calloc(module->function_count, sizeof *out->callees);
    out->code_lengths = calloc(module->function_count, sizeof *out->code_lengths);
    a.locals = calloc(module->function_count, sizeof *a.locals);
    a.results = calloc(module->function_count, sizeof *a.results);
    if (!out->callees || !out->code_lengths || !a.locals || !a.results) goto done;
    /* I retain every global referenced by the module, including late stores. */
    for (uint32_t f = 0; f < module->function_count; ++f) {
        const NvmFunctionEntry *fn = &module->functions[f];
        if (fn->arity > fn->local_count || fn->code_offset > module->code_size ||
            fn->code_length > module->code_size - fn->code_offset) goto done;
        out->code_lengths[f] = fn->code_length;
        out->callees[f] = calloc(fn->code_length ? fn->code_length : 1, sizeof **out->callees);
        a.locals[f] = calloc(fn->local_count ? fn->local_count : 1, sizeof **a.locals);
        if (!out->callees[f] || !a.locals[f]) goto done;
        a.results[f] = fresh(&a);
        for (uint16_t l = 0; l < fn->local_count; ++l) a.locals[f][l] = fresh(&a);
        for (uint32_t pc = 0; pc < fn->code_length;) {
            DecodedInstruction in;
            uint32_t size = isa_decode(module->code + fn->code_offset + pc, fn->code_length - pc, &in);
            if (!size) goto done;
            uint32_t fields = in.opcode == OP_AGG_PACK ? in.operands[3].u16 :
                in.opcode == OP_AGG_GET ? (uint32_t)in.operands[0].u16 + 1 : 0;
            if (fields > a.field_count) a.field_count = fields;
            if (in.opcode == OP_LOAD_GLOBAL || in.opcode == OP_STORE_GLOBAL) {
                if (in.operands[0].u32 >= NVM_MAX_GLOBALS) goto done;
                size_t required = (size_t)in.operands[0].u32 + 1;
                if (required > a.global_count) a.global_count = required;
            }
            pc += size;
        }
    }
    a.globals = calloc(a.global_count ? a.global_count : 1, sizeof *a.globals);
    if (!a.globals) goto done;
    for (size_t i = 0; i < a.global_count; ++i) a.globals[i] = fresh(&a);
    for (uint32_t f = 0; f < module->function_count; ++f)
        if (!collect_function(&a, f)) goto done;
    for (;;) {
        if (!nvm_shape_solve_conversions(&out->shapes)) goto done;
        int changed = 0;
        if (!connect_array_aliases(&a, &changed)) goto done;
        for (size_t c = 0; c < a.call_count; ++c) {
            IndirectCall *call = &a.calls[c];
            if (nvm_shape_kind(&out->shapes, call->callee) != NVM_SHAPE_FUNCTION) continue;
            size_t count = nvm_shape_function_count(&out->shapes, call->callee);
            for (size_t i = 0; i < count; ++i) {
                uint32_t target;
                if (!nvm_shape_function_target(&out->shapes, call->callee, i, &target)) goto done;
                size_t seen = 0;
                while (seen < call->linked_count && call->linked[seen] != target) ++seen;
                if (seen < call->linked_count) continue;
                if (!connect_call(&a, target, call->arguments, call->arity, call->results, call->result)) goto done;
                if (call->linked_count >= SIZE_MAX / sizeof *call->linked) goto done;
                uint32_t *next = realloc(call->linked, (call->linked_count + 1) * sizeof *next);
                if (!next) goto done;
                call->linked = next; call->linked[call->linked_count++] = target;
                changed = 1;
            }
        }
        if (!changed) break;
    }
    ok = 1;
done:
    if (!ok && !out->error[0]) fail(&a, out->shapes.error ? out->shapes.error : "I cannot construct callable constraints");
    if (a.locals) for (uint32_t f = 0; f < module->function_count; ++f) free(a.locals[f]);
    for (size_t i = 0; i < a.call_count; ++i) { free(a.calls[i].arguments); free(a.calls[i].linked); }
    free(a.calls); free(a.locals); free(a.results); free(a.globals);
    return ok;
}
