/*
 * NanoVM - Bytecode execution engine
 *
 * Simple switch dispatch over all NanoISA opcodes.
 */

#include "vm.h"
#include "vm_ffi.h"
#include "cop_protocol.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

/* ========================================================================
 * Error Handling
 * ======================================================================== */

static VmResult vm_error(VmState *vm, VmResult err, const char *fmt, ...) {
    vm->last_error = err;
    va_list args;
    va_start(args, fmt);
    vsnprintf(vm->error_msg, sizeof(vm->error_msg), fmt, args);
    va_end(args);
    return err;
}

const char *vm_error_string(VmResult result) {
    switch (result) {
        case VM_OK:                   return "OK";
        case VM_ERR_STACK_OVERFLOW:   return "Stack overflow";
        case VM_ERR_STACK_UNDERFLOW:  return "Stack underflow";
        case VM_ERR_CALL_DEPTH:       return "Call stack overflow";
        case VM_ERR_INVALID_OPCODE:   return "Invalid opcode";
        case VM_ERR_TYPE_ERROR:       return "Type error";
        case VM_ERR_OUT_OF_BOUNDS:    return "Index out of bounds";
        case VM_ERR_DIV_ZERO:         return "Division by zero";
        case VM_ERR_ASSERT_FAILED:    return "Assertion failed";
        case VM_ERR_UNDEFINED_GLOBAL: return "Undefined global";
        case VM_ERR_UNDEFINED_FUNCTION: return "Undefined function";
        case VM_ERR_NOT_IMPLEMENTED:  return "Not implemented";
        case VM_ERR_MEMORY:           return "Out of memory";
        case VM_ERR_DECODE:           return "Instruction decode error";
    }
    return "Unknown error";
}

/* ========================================================================
 * Init / Destroy
 * ======================================================================== */

void vm_init(VmState *vm, const NvmModule *module) {
    memset(vm, 0, sizeof(*vm));
    vm->module = module;
    vm->stack_capacity = VM_STACK_INITIAL;
    vm->stack = calloc(vm->stack_capacity, sizeof(NanoValue));
    vm->output = NULL; /* default stdout */
    vm->cop_in_fd = -1;
    vm->cop_out_fd = -1;
    vm->cop_pid = -1;
    vm->cop_mailbox = NULL;
    vm->cop_mailbox_size = 0;
    vm->cop_sig_send_fd = -1;
    vm->cop_sig_recv_fd = -1;
    /* Read per-call timeout from env (default 5 s); -1 = unlimited */
    const char *tenv = getenv("COP_TIMEOUT_MS");
    vm->cop_timeout_ms = tenv ? atoi(tenv) : 5000;
    vm_heap_init(&vm->heap);
}

void vm_destroy(VmState *vm) {
    /* Release all globals */
    for (uint32_t i = 0; i < vm->global_count; i++) {
        vm_release(&vm->heap, vm->globals[i]);
    }
    /* Release all stack values */
    for (uint32_t i = 0; i < vm->stack_size; i++) {
        vm_release(&vm->heap, vm->stack[i]);
    }
    free(vm->stack);
    free(vm->linked_modules);
    free(vm->memory);
    vm_heap_destroy(&vm->heap);
}

bool vm_memory_resize(VmState *vm, uint64_t size) {
    if (!vm || size > SIZE_MAX) return false;
    if (size == 0) {
        free(vm->memory);
        vm->memory = NULL;
        vm->memory_size = 0;
        return true;
    }
    uint8_t *memory = realloc(vm->memory, (size_t)size);
    if (!memory) return false;
    if (size > vm->memory_size)
        memset(memory + vm->memory_size, 0, (size_t)(size - vm->memory_size));
    vm->memory = memory;
    vm->memory_size = size;
    return true;
}

uint32_t vm_link_module(VmState *vm, const NvmModule *mod) {
    if (!mod) return (uint32_t)-1;
    if (vm->linked_module_count >= vm->linked_module_capacity) {
        uint32_t new_cap = vm->linked_module_capacity ? vm->linked_module_capacity * 2 : 8;
        const NvmModule **new_arr = realloc(vm->linked_modules,
                                            new_cap * sizeof(const NvmModule *));
        if (!new_arr) return (uint32_t)-1;
        vm->linked_modules = new_arr;
        vm->linked_module_capacity = new_cap;
    }
    uint32_t idx = vm->linked_module_count++;
    vm->linked_modules[idx] = mod;
    return idx;
}

/* ========================================================================
 * Stack Operations
 * ======================================================================== */

static inline VmResult stack_push(VmState *vm, NanoValue v) {
    if (vm->stack_size >= vm->stack_capacity) {
        uint32_t new_cap = vm->stack_capacity * 2;
        NanoValue *new_stack = realloc(vm->stack, new_cap * sizeof(NanoValue));
        if (!new_stack) return vm_error(vm, VM_ERR_MEMORY, "Stack grow failed");
        vm->stack = new_stack;
        vm->stack_capacity = new_cap;
    }
    vm->stack[vm->stack_size++] = v;
    return VM_OK;
}

static inline NanoValue stack_pop(VmState *vm) {
    if (vm->stack_size == 0) return val_void();
    return vm->stack[--vm->stack_size];
}

static inline NanoValue stack_peek(VmState *vm, uint32_t offset) {
    if (offset >= vm->stack_size) return val_void();
    return vm->stack[vm->stack_size - 1 - offset];
}

static inline void profile_instruction(VmState *vm, uint8_t opcode) {
    VmProfile *p = &vm->profile;
    if (!p->enabled) return;

    p->retired++;
    p->opcode_counts[opcode]++;
    if (p->has_previous) {
        uint16_t pair = (uint16_t)(((uint16_t)p->previous << 8) | opcode);
        p->pair_counts[pair]++;
        if (p->has_previous_pair) {
            uint32_t key = ((uint32_t)p->previous_pair << 8) | opcode;
            uint32_t slot = (key * 2654435761U) % VM_PROFILE_TRIPLES;
            for (uint32_t probe = 0; probe < VM_PROFILE_TRIPLES; probe++) {
                VmProfileTriple *entry = &p->triples[slot];
                if (!entry->used || entry->key == key) {
                    entry->used = true;
                    entry->key = key;
                    entry->count++;
                    break;
                }
                slot = (slot + 1) % VM_PROFILE_TRIPLES;
            }
        }
        p->previous_pair = pair;
        p->has_previous_pair = true;
    }
    p->previous = opcode;
    p->has_previous = true;
    if (vm->stack_size > p->max_stack_depth)
        p->max_stack_depth = vm->stack_size;
    if (vm->frame_count > p->max_frame_depth)
        p->max_frame_depth = vm->frame_count;
}

void vm_profile_enable(VmState *vm, bool enabled) {
    if (!vm) return;
    memset(&vm->profile, 0, sizeof(vm->profile));
    vm->profile.enabled = enabled;
}

bool vm_profile_write_json(const VmState *vm, FILE *out) {
    if (!vm || !out) return false;
    const VmProfile *p = &vm->profile;
    fprintf(out, "{\n  \"schema\": \"nanoisa.profile.v1\",\n");
    fprintf(out, "  \"retired\": %llu,\n", (unsigned long long)p->retired);
    fprintf(out, "  \"max_stack_depth\": %u,\n", p->max_stack_depth);
    fprintf(out, "  \"max_frame_depth\": %u,\n", p->max_frame_depth);
    fprintf(out, "  \"branches\": %llu,\n", (unsigned long long)p->branches);
    fprintf(out, "  \"branches_taken\": %llu,\n",
            (unsigned long long)p->branches_taken);
    fprintf(out, "  \"calls\": {\"direct\": %llu, \"indirect\": %llu, "
                 "\"extern\": %llu, \"module\": %llu},\n",
            (unsigned long long)p->direct_calls,
            (unsigned long long)p->indirect_calls,
            (unsigned long long)p->extern_calls,
            (unsigned long long)p->module_calls);
    fprintf(out, "  \"traps\": %llu,\n", (unsigned long long)p->traps);
    fprintf(out, "  \"heap\": {\"allocation_calls\": %llu, "
                 "\"allocated_bytes\": %llu, \"freed_bytes\": %llu, "
                 "\"live_objects\": %llu, \"retain_calls\": %llu, "
                 "\"release_calls\": %llu},\n",
            (unsigned long long)vm->heap.stats.allocation_calls,
            (unsigned long long)vm->heap.stats.allocated,
            (unsigned long long)vm->heap.stats.freed,
            (unsigned long long)vm->heap.stats.num_objects,
            (unsigned long long)vm->heap.stats.retain_calls,
            (unsigned long long)vm->heap.stats.release_calls);
    fprintf(out, "  \"ffi\": {\"request_bytes\": %llu, "
                 "\"response_bytes\": %llu, \"elapsed_ns\": %llu, "
                 "\"failures\": %llu},\n",
            (unsigned long long)p->ffi_request_bytes,
            (unsigned long long)p->ffi_response_bytes,
            (unsigned long long)p->ffi_elapsed_ns,
            (unsigned long long)p->ffi_failures);

    fprintf(out, "  \"opcodes\": {");
    bool first = true;
    for (uint32_t i = 0; i < 256; i++) {
        if (p->opcode_counts[i] == 0) continue;
        fprintf(out, "%s\n    \"0x%02x\": %llu", first ? "" : ",", i,
                (unsigned long long)p->opcode_counts[i]);
        first = false;
    }
    fprintf(out, "%s\n  },\n", first ? "" : "");

    fprintf(out, "  \"pairs\": {");
    first = true;
    for (uint32_t i = 0; i < 256U * 256U; i++) {
        if (p->pair_counts[i] == 0) continue;
        fprintf(out, "%s\n    \"%02x-%02x\": %llu", first ? "" : ",",
                i >> 8, i & 0xff,
                (unsigned long long)p->pair_counts[i]);
        first = false;
    }
    fprintf(out, "%s\n  },\n", first ? "" : "");

    fprintf(out, "  \"triples\": {");
    first = true;
    for (uint32_t i = 0; i < VM_PROFILE_TRIPLES; i++) {
        const VmProfileTriple *entry = &p->triples[i];
        if (!entry->used) continue;
        fprintf(out, "%s\n    \"%02x-%02x-%02x\": %llu",
                first ? "" : ",", (entry->key >> 16) & 0xff,
                (entry->key >> 8) & 0xff, entry->key & 0xff,
                (unsigned long long)entry->count);
        first = false;
    }
    fprintf(out, "%s\n  }\n}\n", first ? "" : "");
    return !ferror(out);
}

/* ========================================================================
 * Helper: output stream
 * ======================================================================== */

static inline FILE *vm_out(VmState *vm) {
    return vm->output ? vm->output : stdout;
}

/* ========================================================================
 * Execution Engine
 * ======================================================================== */

/* Resolve the string pool for convenience */
static inline const char *str_at(const VmState *vm, uint32_t idx) {
    return nvm_get_string(vm->module, idx);
}

NanoValue vm_get_result(VmState *vm) {
    if (vm->stack_size == 0) return val_void();
    return vm->stack[vm->stack_size - 1];
}

/* ========================================================================
 * Trap helpers
 * ======================================================================== */

static inline VmTrap trap_none(void) {
    return (VmTrap){ .type = TRAP_NONE };
}

static inline VmTrap trap_halt(void) {
    return (VmTrap){ .type = TRAP_HALT };
}

static inline VmTrap trap_error(VmState *vm, VmResult err, const char *fmt, ...) {
    vm->last_error = err;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(vm->error_msg, sizeof(vm->error_msg), fmt, ap);
    va_end(ap);
    VmTrap t = { .type = TRAP_ERROR };
    t.data.error.code = err;
    return t;
}

/* ========================================================================
 * Core Execution Engine (the "processor")
 *
 * Runs pure NanoISA instructions.  Returns a VmTrap when it hits an
 * external operation (I/O, FFI, halt) or completes / errors.
 * ======================================================================== */

VmTrap vm_core_execute(VmState *vm) {
    const uint8_t *code = vm->module->code;

    /* Derive code_end from current function */
    const NvmFunctionEntry *cur_fn = &vm->module->functions[vm->current_fn];
    uint32_t code_end = cur_fn->code_offset + cur_fn->code_length;

    VmCallFrame *frame = &vm->frames[vm->frame_count - 1];

    /* Main dispatch loop */
    while (vm->ip < code_end) {
        DecodedInstruction instr;
        uint32_t consumed = isa_decode(code + vm->ip, code_end - vm->ip, &instr);
        if (consumed == 0) {
            return trap_error(vm, VM_ERR_DECODE, "Bad instruction at offset %u", vm->ip);
        }

        uint32_t instr_start = vm->ip;
        vm->ip += consumed;
        profile_instruction(vm, instr.opcode);

        switch (instr.opcode) {

        /* ============================================================
         * Stack & Constants
         * ============================================================ */

        case OP_NOP:
            break;

        case OP_PUSH_I64:
            stack_push(vm, val_int(instr.operands[0].i64));
            break;

        case OP_PUSH_F64:
            stack_push(vm, val_float(instr.operands[0].f64));
            break;

        case OP_PUSH_BOOL:
            stack_push(vm, val_bool(instr.operands[0].u8 != 0));
            break;

        case OP_PUSH_STR: {
            uint32_t idx = instr.operands[0].u32;
            const char *s = str_at(vm, idx);
            if (!s) s = "";
            VmString *vs = vm_string_new(&vm->heap, s, (uint32_t)strlen(s));
            stack_push(vm, val_string(vs));
            break;
        }

        case OP_PUSH_VOID:
            stack_push(vm, val_void());
            break;

        case OP_PUSH_U8:
            stack_push(vm, val_u8(instr.operands[0].u8));
            break;

        case OP_FUNCREF:
            stack_push(vm, val_function(instr.operands[0].u32));
            break;

        case OP_DUP: {
            NanoValue top = stack_peek(vm, 0);
            vm_retain(&vm->heap, top);
            stack_push(vm, top);
            break;
        }

        case OP_POP: {
            NanoValue v = stack_pop(vm);
            vm_release(&vm->heap, v);
            break;
        }

        case OP_SWAP: {
            if (vm->stack_size < 2) break;
            NanoValue a = vm->stack[vm->stack_size - 1];
            NanoValue b = vm->stack[vm->stack_size - 2];
            vm->stack[vm->stack_size - 1] = b;
            vm->stack[vm->stack_size - 2] = a;
            break;
        }

        case OP_ROT3: {
            if (vm->stack_size < 3) break;
            uint32_t top = vm->stack_size - 1;
            NanoValue a = vm->stack[top];
            vm->stack[top] = vm->stack[top - 1];
            vm->stack[top - 1] = vm->stack[top - 2];
            vm->stack[top - 2] = a;
            break;
        }

        case OP_PICK: {
            uint16_t depth = instr.operands[0].u16;
            if (depth >= vm->stack_size)
                return trap_error(vm, VM_ERR_STACK_UNDERFLOW,
                                  "PICK depth %u exceeds stack depth %u",
                                  depth, vm->stack_size);
            NanoValue value = vm->stack[vm->stack_size - 1 - depth];
            vm_retain(&vm->heap, value);
            stack_push(vm, value);
            break;
        }

        case OP_ROLL: {
            uint16_t depth = instr.operands[0].u16;
            if (depth >= vm->stack_size)
                return trap_error(vm, VM_ERR_STACK_UNDERFLOW,
                                  "ROLL depth %u exceeds stack depth %u",
                                  depth, vm->stack_size);
            uint32_t index = vm->stack_size - 1 - depth;
            NanoValue value = vm->stack[index];
            memmove(&vm->stack[index], &vm->stack[index + 1],
                    depth * sizeof(NanoValue));
            vm->stack[vm->stack_size - 1] = value;
            break;
        }

        /* ============================================================
         * Variable Access
         * ============================================================ */

        case OP_LOAD_LOCAL: {
            uint16_t idx = instr.operands[0].u16;
            uint32_t abs_idx = frame->stack_base + idx;
            if (abs_idx >= vm->stack_size) {
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS, "Local %u out of range", idx);
            }
            NanoValue v = vm->stack[abs_idx];
            vm_retain(&vm->heap, v);
            stack_push(vm, v);
            break;
        }

        case OP_STORE_LOCAL: {
            uint16_t idx = instr.operands[0].u16;
            uint32_t abs_idx = frame->stack_base + idx;
            if (abs_idx >= vm->stack_size) {
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS, "Local %u out of range", idx);
            }
            NanoValue v = stack_pop(vm);
            vm_release(&vm->heap, vm->stack[abs_idx]);
            vm->stack[abs_idx] = v;
            break;
        }

        case OP_LOAD_GLOBAL: {
            uint32_t idx = instr.operands[0].u32;
            if (idx >= VM_MAX_GLOBALS) {
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS, "Global %u out of range", idx);
            }
            NanoValue v = vm->globals[idx];
            vm_retain(&vm->heap, v);
            stack_push(vm, v);
            break;
        }

        case OP_STORE_GLOBAL: {
            uint32_t idx = instr.operands[0].u32;
            if (idx >= VM_MAX_GLOBALS) {
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS, "Global %u out of range", idx);
            }
            NanoValue v = stack_pop(vm);
            vm_release(&vm->heap, vm->globals[idx]);
            vm->globals[idx] = v;
            if (idx >= vm->global_count) vm->global_count = idx + 1;
            break;
        }

        case OP_LOAD_UPVALUE: {
            uint16_t idx = instr.operands[1].u16;
            /* depth (operands[0]) is always 0: codegen flattens upvalue chains so
             * every captured variable lives in the immediate closure's capture array. */
            if (frame->closure && idx < frame->closure->capture_count) {
                NanoValue v = frame->closure->captures[idx];
                vm_retain(&vm->heap, v);
                stack_push(vm, v);
            } else {
                stack_push(vm, val_void());
            }
            break;
        }

        case OP_STORE_UPVALUE: {
            uint16_t idx = instr.operands[1].u16;
            /* depth (operands[0]) always 0 — see OP_LOAD_UPVALUE note above */
            NanoValue v = stack_pop(vm);
            if (frame->closure && idx < frame->closure->capture_count) {
                vm_release(&vm->heap, frame->closure->captures[idx]);
                frame->closure->captures[idx] = v;
            } else {
                vm_release(&vm->heap, v);
            }
            break;
        }

        /* ============================================================
         * Arithmetic
         * ============================================================ */

        case OP_ARRAY_ADD:
            if (stack_peek(vm, 0).tag != TAG_ARRAY
                    && stack_peek(vm, 1).tag != TAG_ARRAY)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "ARRAY_ADD requires at least one array");
            goto dynamic_add;
        case OP_ADD: {
dynamic_add:
            ;
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            /* Coerce enum to int for arithmetic */
            if (a.tag == TAG_ENUM) { a = val_int((int64_t)a.as.enum_val); }
            if (b.tag == TAG_ENUM) { b = val_int((int64_t)b.as.enum_val); }
            if (a.tag == TAG_INT && b.tag == TAG_INT) {
                stack_push(vm, val_int(a.as.i64 + b.as.i64));
            } else if (a.tag == TAG_FLOAT && b.tag == TAG_FLOAT) {
                stack_push(vm, val_float(a.as.f64 + b.as.f64));
            } else if (a.tag == TAG_FLOAT && b.tag == TAG_INT) {
                stack_push(vm, val_float(a.as.f64 + (double)b.as.i64));
            } else if (a.tag == TAG_INT && b.tag == TAG_FLOAT) {
                stack_push(vm, val_float((double)a.as.i64 + b.as.f64));
            } else if (a.tag == TAG_STRING && b.tag == TAG_STRING) {
                VmString *s = vm_string_concat(&vm->heap, a.as.string, b.as.string);
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                stack_push(vm, val_string(s));
            } else if (a.tag == TAG_ARRAY && b.tag == TAG_ARRAY) {
                /* Element-wise array addition (supports int, float, string) */
                VmArray *arr_a = a.as.array;
                VmArray *arr_b = b.as.array;
                uint32_t len = arr_a && arr_b ?
                    (arr_a->length < arr_b->length ? arr_a->length : arr_b->length) : 0;
                VmArray *result = vm_array_new(&vm->heap, TAG_INT, len);
                for (uint32_t ai = 0; ai < len; ai++) {
                    NanoValue ea = arr_a->elements[ai];
                    NanoValue eb = arr_b->elements[ai];
                    NanoValue ev;
                    if (ea.tag == TAG_STRING && eb.tag == TAG_STRING) {
                        VmString *s = vm_string_concat(&vm->heap, ea.as.string, eb.as.string);
                        ev = val_string(s);
                    } else if (ea.tag == TAG_INT && eb.tag == TAG_INT)
                        ev = val_int(ea.as.i64 + eb.as.i64);
                    else if (ea.tag == TAG_FLOAT && eb.tag == TAG_FLOAT)
                        ev = val_float(ea.as.f64 + eb.as.f64);
                    else if (ea.tag == TAG_FLOAT && eb.tag == TAG_INT)
                        ev = val_float(ea.as.f64 + (double)eb.as.i64);
                    else if (ea.tag == TAG_INT && eb.tag == TAG_FLOAT)
                        ev = val_float((double)ea.as.i64 + eb.as.f64);
                    else
                        ev = val_int(ea.as.i64 + eb.as.i64);
                    vm_array_push(&vm->heap, result, ev);
                }
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                NanoValue rv = {0};
                rv.tag = TAG_ARRAY;
                rv.as.array = result;
                stack_push(vm, rv);
            } else if ((a.tag == TAG_ARRAY &&
                        (b.tag == TAG_INT || b.tag == TAG_FLOAT || b.tag == TAG_STRING)) ||
                       ((a.tag == TAG_INT || a.tag == TAG_FLOAT || a.tag == TAG_STRING) &&
                        b.tag == TAG_ARRAY)) {
                /* Scalar broadcast: array + scalar or scalar + array */
                VmArray *arr = (a.tag == TAG_ARRAY) ? a.as.array : b.as.array;
                NanoValue scalar = (a.tag == TAG_ARRAY) ? b : a;
                uint32_t len = arr ? arr->length : 0;
                VmArray *result = vm_array_new(&vm->heap, TAG_INT, len);
                for (uint32_t ai = 0; ai < len; ai++) {
                    NanoValue ea = arr->elements[ai];
                    NanoValue ev;
                    if (ea.tag == TAG_STRING && scalar.tag == TAG_STRING) {
                        /* String concat broadcast */
                        if (a.tag == TAG_ARRAY) {
                            VmString *s = vm_string_concat(&vm->heap, ea.as.string, scalar.as.string);
                            ev = val_string(s);
                        } else {
                            VmString *s = vm_string_concat(&vm->heap, scalar.as.string, ea.as.string);
                            ev = val_string(s);
                        }
                    } else if (ea.tag == TAG_INT && scalar.tag == TAG_INT)
                        ev = val_int(ea.as.i64 + scalar.as.i64);
                    else if (ea.tag == TAG_FLOAT || scalar.tag == TAG_FLOAT) {
                        double da = ea.tag == TAG_FLOAT ? ea.as.f64 : (double)ea.as.i64;
                        double ds = scalar.tag == TAG_FLOAT ? scalar.as.f64 : (double)scalar.as.i64;
                        ev = val_float(da + ds);
                    } else
                        ev = val_int(ea.as.i64 + scalar.as.i64);
                    vm_array_push(&vm->heap, result, ev);
                }
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                NanoValue rv = {0};
                rv.tag = TAG_ARRAY;
                rv.as.array = result;
                stack_push(vm, rv);
            } else {
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "ADD: incompatible types %s + %s",
                                isa_tag_name(a.tag), isa_tag_name(b.tag));
            }
            break;
        }

        case OP_ARRAY_SUB:
            if (stack_peek(vm, 0).tag != TAG_ARRAY
                    && stack_peek(vm, 1).tag != TAG_ARRAY)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "ARRAY_SUB requires at least one array");
            goto dynamic_sub;
        case OP_SUB: {
dynamic_sub:
            ;
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_ENUM) { a = val_int((int64_t)a.as.enum_val); }
            if (b.tag == TAG_ENUM) { b = val_int((int64_t)b.as.enum_val); }
            if (a.tag == TAG_INT && b.tag == TAG_INT) {
                stack_push(vm, val_int(a.as.i64 - b.as.i64));
            } else if (a.tag == TAG_FLOAT && b.tag == TAG_FLOAT) {
                stack_push(vm, val_float(a.as.f64 - b.as.f64));
            } else if (a.tag == TAG_FLOAT && b.tag == TAG_INT) {
                stack_push(vm, val_float(a.as.f64 - (double)b.as.i64));
            } else if (a.tag == TAG_INT && b.tag == TAG_FLOAT) {
                stack_push(vm, val_float((double)a.as.i64 - b.as.f64));
            } else if (a.tag == TAG_ARRAY && b.tag == TAG_ARRAY) {
                VmArray *arr_a = a.as.array;
                VmArray *arr_b = b.as.array;
                uint32_t len = arr_a && arr_b ?
                    (arr_a->length < arr_b->length ? arr_a->length : arr_b->length) : 0;
                VmArray *result = vm_array_new(&vm->heap, TAG_INT, len);
                for (uint32_t ai = 0; ai < len; ai++) {
                    NanoValue ea = arr_a->elements[ai];
                    NanoValue eb = arr_b->elements[ai];
                    NanoValue ev;
                    if (ea.tag == TAG_INT && eb.tag == TAG_INT)
                        ev = val_int(ea.as.i64 - eb.as.i64);
                    else if (ea.tag == TAG_FLOAT || eb.tag == TAG_FLOAT) {
                        double da = ea.tag == TAG_FLOAT ? ea.as.f64 : (double)ea.as.i64;
                        double db = eb.tag == TAG_FLOAT ? eb.as.f64 : (double)eb.as.i64;
                        ev = val_float(da - db);
                    } else
                        ev = val_int(ea.as.i64 - eb.as.i64);
                    vm_array_push(&vm->heap, result, ev);
                }
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                NanoValue rv = {0};
                rv.tag = TAG_ARRAY;
                rv.as.array = result;
                stack_push(vm, rv);
            } else if ((a.tag == TAG_ARRAY && (b.tag == TAG_INT || b.tag == TAG_FLOAT)) ||
                       ((a.tag == TAG_INT || a.tag == TAG_FLOAT) && b.tag == TAG_ARRAY)) {
                VmArray *arr = (a.tag == TAG_ARRAY) ? a.as.array : b.as.array;
                NanoValue scalar = (a.tag == TAG_ARRAY) ? b : a;
                bool arr_is_left = (a.tag == TAG_ARRAY);
                uint32_t len = arr ? arr->length : 0;
                VmArray *result = vm_array_new(&vm->heap, TAG_INT, len);
                for (uint32_t ai = 0; ai < len; ai++) {
                    NanoValue ea = arr->elements[ai];
                    NanoValue ev;
                    double da = ea.tag == TAG_FLOAT ? ea.as.f64 : (double)ea.as.i64;
                    double ds = scalar.tag == TAG_FLOAT ? scalar.as.f64 : (double)scalar.as.i64;
                    double dr = arr_is_left ? da - ds : ds - da;
                    if (ea.tag == TAG_INT && scalar.tag == TAG_INT)
                        ev = val_int(arr_is_left ? ea.as.i64 - scalar.as.i64 : scalar.as.i64 - ea.as.i64);
                    else
                        ev = val_float(dr);
                    vm_array_push(&vm->heap, result, ev);
                }
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                NanoValue rv = {0};
                rv.tag = TAG_ARRAY;
                rv.as.array = result;
                stack_push(vm, rv);
            } else {
                return trap_error(vm, VM_ERR_TYPE_ERROR, "SUB: type error");
            }
            break;
        }

        case OP_ARRAY_MUL:
            if (stack_peek(vm, 0).tag != TAG_ARRAY
                    && stack_peek(vm, 1).tag != TAG_ARRAY)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "ARRAY_MUL requires at least one array");
            goto dynamic_mul;
        case OP_MUL: {
dynamic_mul:
            ;
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_ENUM) { a = val_int((int64_t)a.as.enum_val); }
            if (b.tag == TAG_ENUM) { b = val_int((int64_t)b.as.enum_val); }
            if (a.tag == TAG_INT && b.tag == TAG_INT) {
                stack_push(vm, val_int(a.as.i64 * b.as.i64));
            } else if (a.tag == TAG_FLOAT && b.tag == TAG_FLOAT) {
                stack_push(vm, val_float(a.as.f64 * b.as.f64));
            } else if (a.tag == TAG_FLOAT && b.tag == TAG_INT) {
                stack_push(vm, val_float(a.as.f64 * (double)b.as.i64));
            } else if (a.tag == TAG_INT && b.tag == TAG_FLOAT) {
                stack_push(vm, val_float((double)a.as.i64 * b.as.f64));
            } else if (a.tag == TAG_ARRAY && b.tag == TAG_ARRAY) {
                VmArray *arr_a = a.as.array;
                VmArray *arr_b = b.as.array;
                uint32_t len = arr_a && arr_b ?
                    (arr_a->length < arr_b->length ? arr_a->length : arr_b->length) : 0;
                VmArray *result = vm_array_new(&vm->heap, TAG_INT, len);
                for (uint32_t ai = 0; ai < len; ai++) {
                    NanoValue ea = arr_a->elements[ai];
                    NanoValue eb = arr_b->elements[ai];
                    NanoValue ev;
                    if (ea.tag == TAG_INT && eb.tag == TAG_INT)
                        ev = val_int(ea.as.i64 * eb.as.i64);
                    else if (ea.tag == TAG_FLOAT || eb.tag == TAG_FLOAT) {
                        double da = ea.tag == TAG_FLOAT ? ea.as.f64 : (double)ea.as.i64;
                        double db = eb.tag == TAG_FLOAT ? eb.as.f64 : (double)eb.as.i64;
                        ev = val_float(da * db);
                    } else
                        ev = val_int(ea.as.i64 * eb.as.i64);
                    vm_array_push(&vm->heap, result, ev);
                }
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                NanoValue rv = {0};
                rv.tag = TAG_ARRAY;
                rv.as.array = result;
                stack_push(vm, rv);
            } else if ((a.tag == TAG_ARRAY && (b.tag == TAG_INT || b.tag == TAG_FLOAT)) ||
                       ((a.tag == TAG_INT || a.tag == TAG_FLOAT) && b.tag == TAG_ARRAY)) {
                VmArray *arr = (a.tag == TAG_ARRAY) ? a.as.array : b.as.array;
                NanoValue scalar = (a.tag == TAG_ARRAY) ? b : a;
                uint32_t len = arr ? arr->length : 0;
                VmArray *result = vm_array_new(&vm->heap, TAG_INT, len);
                for (uint32_t ai = 0; ai < len; ai++) {
                    NanoValue ea = arr->elements[ai];
                    NanoValue ev;
                    if (ea.tag == TAG_INT && scalar.tag == TAG_INT)
                        ev = val_int(ea.as.i64 * scalar.as.i64);
                    else {
                        double da = ea.tag == TAG_FLOAT ? ea.as.f64 : (double)ea.as.i64;
                        double ds = scalar.tag == TAG_FLOAT ? scalar.as.f64 : (double)scalar.as.i64;
                        ev = val_float(da * ds);
                    }
                    vm_array_push(&vm->heap, result, ev);
                }
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                NanoValue rv = {0};
                rv.tag = TAG_ARRAY;
                rv.as.array = result;
                stack_push(vm, rv);
            } else {
                return trap_error(vm, VM_ERR_TYPE_ERROR, "MUL: type error");
            }
            break;
        }

        case OP_ARRAY_DIV:
            if (stack_peek(vm, 0).tag != TAG_ARRAY
                    && stack_peek(vm, 1).tag != TAG_ARRAY)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "ARRAY_DIV requires at least one array");
            goto dynamic_div;
        case OP_DIV: {
dynamic_div:
            ;
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_ENUM) { a = val_int((int64_t)a.as.enum_val); }
            if (b.tag == TAG_ENUM) { b = val_int((int64_t)b.as.enum_val); }
            if (a.tag == TAG_INT && b.tag == TAG_INT) {
                /* Total division: by zero = 0 (matches Coq semantics); the
                 * INT64_MIN / -1 case is signed-overflow UB that raises SIGFPE
                 * on x86-64, so wrap to INT64_MIN instead of dividing. */
                int64_t q;
                if (b.as.i64 == 0) q = 0;
                else if (a.as.i64 == INT64_MIN && b.as.i64 == -1) q = INT64_MIN;
                else q = a.as.i64 / b.as.i64;
                stack_push(vm, val_int(q));
            } else if (a.tag == TAG_FLOAT && b.tag == TAG_FLOAT) {
                stack_push(vm, val_float(b.as.f64 == 0.0 ? 0.0 : a.as.f64 / b.as.f64));
            } else if (a.tag == TAG_FLOAT && b.tag == TAG_INT) {
                stack_push(vm, val_float(b.as.i64 == 0 ? 0.0 : a.as.f64 / (double)b.as.i64));
            } else if (a.tag == TAG_INT && b.tag == TAG_FLOAT) {
                stack_push(vm, val_float(b.as.f64 == 0.0 ? 0.0 : (double)a.as.i64 / b.as.f64));
            } else if (a.tag == TAG_ARRAY && b.tag == TAG_ARRAY) {
                VmArray *arr_a = a.as.array;
                VmArray *arr_b = b.as.array;
                uint32_t len = arr_a && arr_b ?
                    (arr_a->length < arr_b->length ? arr_a->length : arr_b->length) : 0;
                VmArray *result = vm_array_new(&vm->heap, TAG_INT, len);
                for (uint32_t ai = 0; ai < len; ai++) {
                    NanoValue ea = arr_a->elements[ai];
                    NanoValue eb = arr_b->elements[ai];
                    NanoValue ev;
                    if (ea.tag == TAG_INT && eb.tag == TAG_INT)
                        ev = val_int(eb.as.i64 == 0 ? 0 : ea.as.i64 / eb.as.i64);
                    else {
                        double da = ea.tag == TAG_FLOAT ? ea.as.f64 : (double)ea.as.i64;
                        double db = eb.tag == TAG_FLOAT ? eb.as.f64 : (double)eb.as.i64;
                        ev = val_float(db == 0.0 ? 0.0 : da / db);
                    }
                    vm_array_push(&vm->heap, result, ev);
                }
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                NanoValue rv = {0};
                rv.tag = TAG_ARRAY;
                rv.as.array = result;
                stack_push(vm, rv);
            } else if ((a.tag == TAG_ARRAY && (b.tag == TAG_INT || b.tag == TAG_FLOAT)) ||
                       ((a.tag == TAG_INT || a.tag == TAG_FLOAT) && b.tag == TAG_ARRAY)) {
                VmArray *arr = (a.tag == TAG_ARRAY) ? a.as.array : b.as.array;
                NanoValue scalar = (a.tag == TAG_ARRAY) ? b : a;
                bool arr_is_left = (a.tag == TAG_ARRAY);
                uint32_t len = arr ? arr->length : 0;
                VmArray *result = vm_array_new(&vm->heap, TAG_INT, len);
                for (uint32_t ai = 0; ai < len; ai++) {
                    NanoValue ea = arr->elements[ai];
                    NanoValue ev;
                    double da = ea.tag == TAG_FLOAT ? ea.as.f64 : (double)ea.as.i64;
                    double ds = scalar.tag == TAG_FLOAT ? scalar.as.f64 : (double)scalar.as.i64;
                    if (ea.tag == TAG_INT && scalar.tag == TAG_INT) {
                        if (arr_is_left)
                            ev = val_int(scalar.as.i64 == 0 ? 0 : ea.as.i64 / scalar.as.i64);
                        else
                            ev = val_int(ea.as.i64 == 0 ? 0 : scalar.as.i64 / ea.as.i64);
                    } else {
                        double dr = arr_is_left ? (ds == 0.0 ? 0.0 : da / ds)
                                                : (da == 0.0 ? 0.0 : ds / da);
                        ev = val_float(dr);
                    }
                    vm_array_push(&vm->heap, result, ev);
                }
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                NanoValue rv = {0};
                rv.tag = TAG_ARRAY;
                rv.as.array = result;
                stack_push(vm, rv);
            } else {
                return trap_error(vm, VM_ERR_TYPE_ERROR, "DIV: type error");
            }
            break;
        }

        case OP_MOD: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_INT && b.tag == TAG_INT) {
                /* Total modulo: by zero = 0; INT64_MIN % -1 is mathematically 0
                 * but signed-overflow UB (SIGFPE) if computed directly. */
                int64_t r;
                if (b.as.i64 == 0) r = 0;
                else if (a.as.i64 == INT64_MIN && b.as.i64 == -1) r = 0;
                else r = a.as.i64 % b.as.i64;
                stack_push(vm, val_int(r));
            } else {
                return trap_error(vm, VM_ERR_TYPE_ERROR, "MOD: type error");
            }
            break;
        }

        case OP_NEG: {
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_INT) {
                /* -INT64_MIN overflows (UB); wrap to INT64_MIN. */
                stack_push(vm, val_int(a.as.i64 == INT64_MIN ? INT64_MIN : -a.as.i64));
            } else if (a.tag == TAG_FLOAT) {
                stack_push(vm, val_float(-a.as.f64));
            } else {
                return trap_error(vm, VM_ERR_TYPE_ERROR, "NEG: type error");
            }
            break;
        }

        case OP_I64_ADD:
        case OP_I64_SUB:
        case OP_I64_MUL:
        case OP_I64_DIV_S:
        case OP_I64_REM_S: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_ENUM) a = val_int((int64_t)a.as.enum_val);
            if (b.tag == TAG_ENUM) b = val_int((int64_t)b.as.enum_val);
            if (a.tag != TAG_INT || b.tag != TAG_INT)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires two integers",
                                  isa_get_info(instr.opcode)->name);
            int64_t result = 0;
            if (instr.opcode == OP_I64_ADD) result = a.as.i64 + b.as.i64;
            else if (instr.opcode == OP_I64_SUB) result = a.as.i64 - b.as.i64;
            else if (instr.opcode == OP_I64_MUL) result = a.as.i64 * b.as.i64;
            else if (instr.opcode == OP_I64_DIV_S) {
                if (b.as.i64 == 0) result = 0;
                else if (a.as.i64 == INT64_MIN && b.as.i64 == -1) result = INT64_MIN;
                else result = a.as.i64 / b.as.i64;
            } else {
                if (b.as.i64 == 0 || (a.as.i64 == INT64_MIN && b.as.i64 == -1))
                    result = 0;
                else result = a.as.i64 % b.as.i64;
            }
            stack_push(vm, val_int(result));
            break;
        }

        case OP_I64_NEG: {
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_INT)
                return trap_error(vm, VM_ERR_TYPE_ERROR, "I64_NEG requires an integer");
            stack_push(vm, val_int(a.as.i64 == INT64_MIN ? INT64_MIN : -a.as.i64));
            break;
        }

        case OP_F64_ADD:
        case OP_F64_SUB:
        case OP_F64_MUL:
        case OP_F64_DIV: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_FLOAT || b.tag != TAG_FLOAT)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires two floats",
                                  isa_get_info(instr.opcode)->name);
            double result = 0.0;
            if (instr.opcode == OP_F64_ADD) result = a.as.f64 + b.as.f64;
            else if (instr.opcode == OP_F64_SUB) result = a.as.f64 - b.as.f64;
            else if (instr.opcode == OP_F64_MUL) result = a.as.f64 * b.as.f64;
            else result = b.as.f64 == 0.0 ? 0.0 : a.as.f64 / b.as.f64;
            stack_push(vm, val_float(result));
            break;
        }

        case OP_F64_NEG: {
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_FLOAT)
                return trap_error(vm, VM_ERR_TYPE_ERROR, "F64_NEG requires a float");
            stack_push(vm, val_float(-a.as.f64));
            break;
        }

        /* ============================================================
         * Comparison
         * ============================================================ */

        case OP_EQ: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(val_equal(a, b)));
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            break;
        }

        case OP_NE: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(!val_equal(a, b)));
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            break;
        }

        case OP_LT: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(val_compare(a, b) < 0));
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            break;
        }

        case OP_LE: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(val_compare(a, b) <= 0));
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            break;
        }

        case OP_GT: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(val_compare(a, b) > 0));
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            break;
        }

        case OP_GE: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(val_compare(a, b) >= 0));
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            break;
        }

        case OP_I64_EQ:
        case OP_I64_NE:
        case OP_I64_LT_S:
        case OP_I64_LE_S:
        case OP_I64_GT_S:
        case OP_I64_GE_S: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_ENUM) a = val_int((int64_t)a.as.enum_val);
            if (b.tag == TAG_ENUM) b = val_int((int64_t)b.as.enum_val);
            if (a.tag != TAG_INT || b.tag != TAG_INT)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires two integers",
                                  isa_get_info(instr.opcode)->name);
            bool result = false;
            if (instr.opcode == OP_I64_EQ) result = a.as.i64 == b.as.i64;
            else if (instr.opcode == OP_I64_NE) result = a.as.i64 != b.as.i64;
            else if (instr.opcode == OP_I64_LT_S) result = a.as.i64 < b.as.i64;
            else if (instr.opcode == OP_I64_LE_S) result = a.as.i64 <= b.as.i64;
            else if (instr.opcode == OP_I64_GT_S) result = a.as.i64 > b.as.i64;
            else result = a.as.i64 >= b.as.i64;
            stack_push(vm, val_bool(result));
            break;
        }

        case OP_F64_EQ:
        case OP_F64_NE:
        case OP_F64_LT:
        case OP_F64_LE:
        case OP_F64_GT:
        case OP_F64_GE: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_FLOAT || b.tag != TAG_FLOAT)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires two floats",
                                  isa_get_info(instr.opcode)->name);
            bool result = false;
            if (instr.opcode == OP_F64_EQ) result = a.as.f64 == b.as.f64;
            else if (instr.opcode == OP_F64_NE) result = a.as.f64 != b.as.f64;
            else if (instr.opcode == OP_F64_LT) result = a.as.f64 < b.as.f64;
            else if (instr.opcode == OP_F64_LE) result = a.as.f64 <= b.as.f64;
            else if (instr.opcode == OP_F64_GT) result = a.as.f64 > b.as.f64;
            else result = a.as.f64 >= b.as.f64;
            stack_push(vm, val_bool(result));
            break;
        }

        /* ============================================================
         * Logic
         * ============================================================ */

        case OP_AND: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(val_truthy(a) && val_truthy(b)));
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            break;
        }

        case OP_OR: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(val_truthy(a) || val_truthy(b)));
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            break;
        }

        case OP_NOT: {
            NanoValue a = stack_pop(vm);
            stack_push(vm, val_bool(!val_truthy(a)));
            vm_release(&vm->heap, a);
            break;
        }

        case OP_BOOL_AND:
        case OP_BOOL_OR: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_BOOL || b.tag != TAG_BOOL)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires two booleans",
                                  isa_get_info(instr.opcode)->name);
            stack_push(vm, val_bool(instr.opcode == OP_BOOL_AND
                                    ? a.as.boolean && b.as.boolean
                                    : a.as.boolean || b.as.boolean));
            break;
        }

        case OP_BOOL_NOT: {
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_BOOL)
                return trap_error(vm, VM_ERR_TYPE_ERROR, "BOOL_NOT requires a boolean");
            stack_push(vm, val_bool(!a.as.boolean));
            break;
        }

        case OP_I64_DIV_U:
        case OP_I64_REM_U:
        case OP_I64_LT_U:
        case OP_I64_LE_U:
        case OP_I64_GT_U:
        case OP_I64_GE_U:
        case OP_I64_AND:
        case OP_I64_OR:
        case OP_I64_XOR:
        case OP_I64_SHL:
        case OP_I64_SHR_S:
        case OP_I64_SHR_U: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_INT || b.tag != TAG_INT)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires two integers",
                                  isa_get_info(instr.opcode)->name);
            uint64_t ua = (uint64_t)a.as.i64;
            uint64_t ub = (uint64_t)b.as.i64;
            uint32_t shift = (uint32_t)(ub & 63U);
            switch (instr.opcode) {
                case OP_I64_DIV_U: stack_push(vm, val_int(ub ? (int64_t)(ua / ub) : 0)); break;
                case OP_I64_REM_U: stack_push(vm, val_int(ub ? (int64_t)(ua % ub) : 0)); break;
                case OP_I64_LT_U: stack_push(vm, val_bool(ua < ub)); break;
                case OP_I64_LE_U: stack_push(vm, val_bool(ua <= ub)); break;
                case OP_I64_GT_U: stack_push(vm, val_bool(ua > ub)); break;
                case OP_I64_GE_U: stack_push(vm, val_bool(ua >= ub)); break;
                case OP_I64_AND: stack_push(vm, val_int((int64_t)(ua & ub))); break;
                case OP_I64_OR: stack_push(vm, val_int((int64_t)(ua | ub))); break;
                case OP_I64_XOR: stack_push(vm, val_int((int64_t)(ua ^ ub))); break;
                case OP_I64_SHL: stack_push(vm, val_int((int64_t)(ua << shift))); break;
                case OP_I64_SHR_S: stack_push(vm, val_int(a.as.i64 >> shift)); break;
                case OP_I64_SHR_U: stack_push(vm, val_int((int64_t)(ua >> shift))); break;
                default: break;
            }
            break;
        }

        case OP_I64_INVERT: {
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_INT)
                return trap_error(vm, VM_ERR_TYPE_ERROR, "I64_INVERT requires an integer");
            stack_push(vm, val_int((int64_t)~(uint64_t)a.as.i64));
            break;
        }

        case OP_I64_ADD_CARRY:
        case OP_I64_SUB_BORROW: {
            NanoValue carry = stack_pop(vm);
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_INT || b.tag != TAG_INT || carry.tag != TAG_INT)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires three integers",
                                  isa_get_info(instr.opcode)->name);
            uint64_t ua = (uint64_t)a.as.i64;
            uint64_t ub = (uint64_t)b.as.i64;
            uint64_t uc = (uint64_t)carry.as.i64 & 1U;
            uint64_t low;
            uint64_t high;
            if (instr.opcode == OP_I64_ADD_CARRY) {
                low = ua + ub;
                high = low < ua;
                uint64_t with_carry = low + uc;
                high |= with_carry < low;
                low = with_carry;
            } else {
                low = ua - ub;
                high = ua < ub;
                uint64_t with_borrow = low - uc;
                high |= low < uc;
                low = with_borrow;
            }
            stack_push(vm, val_int((int64_t)low));
            stack_push(vm, val_int((int64_t)high));
            break;
        }

        case OP_I64_MUL_WIDE_S:
        case OP_I64_MUL_WIDE_U: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_INT || b.tag != TAG_INT)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires two integers",
                                  isa_get_info(instr.opcode)->name);
            uint64_t low;
            uint64_t high;
#if defined(__SIZEOF_INT128__)
            if (instr.opcode == OP_I64_MUL_WIDE_S) {
                __int128 product = (__int128)a.as.i64 * (__int128)b.as.i64;
                low = (uint64_t)product;
                high = (uint64_t)((unsigned __int128)product >> 64);
            } else {
                unsigned __int128 product =
                    (unsigned __int128)(uint64_t)a.as.i64
                    * (unsigned __int128)(uint64_t)b.as.i64;
                low = (uint64_t)product;
                high = (uint64_t)(product >> 64);
            }
#else
#error "NanoISA wide multiplication requires compiler 128-bit integer support"
#endif
            stack_push(vm, val_int((int64_t)low));
            stack_push(vm, val_int((int64_t)high));
            break;
        }

        /* ============================================================
         * Control Flow
         * ============================================================ */

        case OP_JMP: {
            if (vm->profile.enabled) {
                vm->profile.branches++;
                vm->profile.branches_taken++;
            }
            int32_t offset = instr.operands[0].i32;
            vm->ip = (uint32_t)((int32_t)instr_start + offset);
            break;
        }

        case OP_JMP_TRUE: {
            NanoValue cond = stack_pop(vm);
            bool taken = val_truthy(cond);
            if (vm->profile.enabled) {
                vm->profile.branches++;
                if (taken) vm->profile.branches_taken++;
            }
            if (taken) {
                vm->ip = (uint32_t)((int32_t)instr_start + instr.operands[0].i32);
            }
            vm_release(&vm->heap, cond);
            break;
        }

        case OP_JMP_FALSE: {
            NanoValue cond = stack_pop(vm);
            bool taken = !val_truthy(cond);
            if (vm->profile.enabled) {
                vm->profile.branches++;
                if (taken) vm->profile.branches_taken++;
            }
            if (taken) {
                vm->ip = (uint32_t)((int32_t)instr_start + instr.operands[0].i32);
            }
            vm_release(&vm->heap, cond);
            break;
        }

        case OP_CALL: {
            if (vm->profile.enabled) vm->profile.direct_calls++;
            uint32_t callee_idx = instr.operands[0].u32;
            if (callee_idx >= vm->module->function_count) {
                return trap_error(vm, VM_ERR_UNDEFINED_FUNCTION, "Function %u not found", callee_idx);
            }

            const NvmFunctionEntry *callee = &vm->module->functions[callee_idx];

            if (vm->frame_count >= VM_MAX_FRAMES) {
                return trap_error(vm, VM_ERR_CALL_DEPTH, "Call depth exceeded");
            }

            /* Arguments are already on the stack, pop them into the new frame */
            uint32_t new_base = vm->stack_size - callee->arity;

            /* Allocate space for remaining locals */
            for (uint16_t i = callee->arity; i < callee->local_count; i++) {
                stack_push(vm, val_void());
            }

            VmCallFrame *new_frame = &vm->frames[vm->frame_count++];
            new_frame->fn_idx = callee_idx;
            new_frame->return_ip = vm->ip;
            new_frame->stack_base = new_base;
            new_frame->local_count = callee->local_count;
            new_frame->closure = NULL;
            new_frame->module = vm->module;
            new_frame->current_line = 0;
            new_frame->current_col  = 0;

            /* Save current execution context */
            frame = new_frame;
            vm->current_fn = callee_idx;
            vm->ip = callee->code_offset;
            code_end = callee->code_offset + callee->code_length;
            break;
        }

        case OP_CALL_INDIRECT: {
            if (vm->profile.enabled) vm->profile.indirect_calls++;
            NanoValue fn_val = stack_pop(vm);
            if (fn_val.tag == TAG_FUNCTION || fn_val.tag == TAG_CLOSURE) {
                VmClosure *closure = NULL;
                uint32_t callee_idx;

                if (fn_val.tag == TAG_CLOSURE) {
                    closure = fn_val.as.closure;
                    callee_idx = closure->fn_idx;
                } else {
                    callee_idx = fn_val.as.fn_idx;
                }

                if (callee_idx >= vm->module->function_count) {
                    return trap_error(vm, VM_ERR_UNDEFINED_FUNCTION, "Indirect call: fn %u not found", callee_idx);
                }

                const NvmFunctionEntry *callee = &vm->module->functions[callee_idx];

                if (vm->frame_count >= VM_MAX_FRAMES) {
                    return trap_error(vm, VM_ERR_CALL_DEPTH, "Call depth exceeded");
                }

                uint32_t new_base = vm->stack_size - callee->arity;
                for (uint16_t i = callee->arity; i < callee->local_count; i++) {
                    stack_push(vm, val_void());
                }

                VmCallFrame *new_frame = &vm->frames[vm->frame_count++];
                new_frame->fn_idx = callee_idx;
                new_frame->return_ip = vm->ip;
                new_frame->stack_base = new_base;
                new_frame->local_count = callee->local_count;
                new_frame->closure = closure;
                new_frame->module = vm->module;
            new_frame->current_line = 0;
            new_frame->current_col  = 0;

                frame = new_frame;
                vm->current_fn = callee_idx;
                vm->ip = callee->code_offset;
                code_end = callee->code_offset + callee->code_length;
            } else {
                return trap_error(vm, VM_ERR_TYPE_ERROR, "CALL_INDIRECT: not a function");
            }
            break;
        }

        case OP_RET: {
            /* Pop the return value (if stack has values above frame locals) */
            NanoValue result = val_void();
            if (vm->stack_size > frame->stack_base + frame->local_count) {
                result = stack_pop(vm);
            }

            /* Clean up locals */
            while (vm->stack_size > frame->stack_base) {
                NanoValue v = stack_pop(vm);
                vm_release(&vm->heap, v);
            }

            /* Save the returning function's return_ip (points to instruction
             * after the CALL in the caller) before we pop the frame */
            uint32_t ret_ip = frame->return_ip;

            vm->frame_count--;

            if (vm->frame_count == 0) {
                /* Return from top-level function - push result and exit */
                stack_push(vm, result);
                return trap_none();
            }

            /* Restore caller's frame context, but use callee's return_ip */
            frame = &vm->frames[vm->frame_count - 1];
            vm->current_fn = frame->fn_idx;
            vm->ip = ret_ip;
            vm->module = frame->module;  /* Restore caller's module */
            code = vm->module->code;     /* Re-derive code pointer */
            const NvmFunctionEntry *caller_fn = &vm->module->functions[frame->fn_idx];
            code_end = caller_fn->code_offset + caller_fn->code_length;

            /* Push return value for caller */
            stack_push(vm, result);
            break;
        }

        case OP_CALL_EXTERN: {
            if (vm->profile.enabled) {
                vm->profile.extern_calls++;
                vm->profile.traps++;
            }
            uint32_t import_idx = instr.operands[0].u32;

            /* Determine arg count from import table */
            if (import_idx >= vm->module->import_count) {
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS,
                                "Import index %u out of range", import_idx);
            }
            int ext_argc = vm->module->imports[import_idx].param_count;

            /* Pop arguments from stack (they were pushed left-to-right,
             * so pop in reverse to get them in order) */
            VmTrap t = { .type = TRAP_EXTERN_CALL };
            t.data.extern_call.import_idx = import_idx;
            t.data.extern_call.argc = ext_argc > 16 ? 16 : ext_argc;
            for (int i = t.data.extern_call.argc - 1; i >= 0; i--) {
                t.data.extern_call.args[i] = stack_pop(vm);
            }
            return t;
        }

        case OP_CALL_MODULE: {
            if (vm->profile.enabled) vm->profile.module_calls++;
            uint32_t mod_idx = instr.operands[0].u32;
            uint32_t fn_idx_m = instr.operands[1].u32;

            /* Bounds check module index */
            if (mod_idx >= vm->linked_module_count) {
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS,
                    "Module index %u out of range (have %u)", mod_idx, vm->linked_module_count);
            }
            const NvmModule *target = vm->linked_modules[mod_idx];

            /* Bounds check function index in target module */
            if (fn_idx_m >= target->function_count) {
                return trap_error(vm, VM_ERR_UNDEFINED_FUNCTION,
                    "Function %u not found in module %u", fn_idx_m, mod_idx);
            }
            const NvmFunctionEntry *callee = &target->functions[fn_idx_m];

            if (vm->frame_count >= VM_MAX_FRAMES) {
                return trap_error(vm, VM_ERR_CALL_DEPTH, "Call depth exceeded");
            }

            uint32_t new_base = vm->stack_size - callee->arity;
            for (uint16_t i = callee->arity; i < callee->local_count; i++) {
                stack_push(vm, val_void());
            }

            /* Create frame for the callee in the target module.
             * frame->module stores the module the callee runs in, so that
             * OP_RET can correctly restore vm->module to the caller's module. */
            VmCallFrame *new_frame = &vm->frames[vm->frame_count++];
            new_frame->fn_idx = fn_idx_m;
            new_frame->return_ip = vm->ip;
            new_frame->stack_base = new_base;
            new_frame->local_count = callee->local_count;
            new_frame->closure = NULL;
            new_frame->module = target;  /* This frame runs in the target module */
            new_frame->current_line = 0;
            new_frame->current_col  = 0;

            /* Switch to target module */
            vm->module = target;
            code = target->code;
            frame = new_frame;
            vm->current_fn = fn_idx_m;
            vm->ip = callee->code_offset;
            code_end = callee->code_offset + callee->code_length;
            break;
        }

        /* ============================================================
         * String Ops
         * ============================================================ */

        case OP_STR_LEN: {
            NanoValue s = stack_pop(vm);
            if (s.tag != TAG_STRING) {
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_LEN: not a string");
            }
            int64_t len = vmstring_len(s.as.string);
            vm_release(&vm->heap, s);
            stack_push(vm, val_int(len));
            break;
        }

        case OP_STR_CONCAT: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_STRING || b.tag != TAG_STRING) {
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_CONCAT: not strings");
            }
            VmString *result = vm_string_concat(&vm->heap, a.as.string, b.as.string);
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            stack_push(vm, val_string(result));
            break;
        }

        case OP_STR_SUBSTR: {
            NanoValue len_v = stack_pop(vm);
            NanoValue start_v = stack_pop(vm);
            NanoValue s = stack_pop(vm);
            if (s.tag != TAG_STRING) {
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_SUBSTR: not a string");
            }
            uint32_t start = (uint32_t)(start_v.tag == TAG_INT ? start_v.as.i64 : 0);
            uint32_t len = (uint32_t)(len_v.tag == TAG_INT ? len_v.as.i64 : 0);
            VmString *result = vm_string_substr(&vm->heap, s.as.string, start, len);
            vm_release(&vm->heap, s);
            stack_push(vm, val_string(result));
            break;
        }

        case OP_STR_CONTAINS: {
            NanoValue needle = stack_pop(vm);
            NanoValue haystack = stack_pop(vm);
            if (haystack.tag != TAG_STRING || needle.tag != TAG_STRING) {
                vm_release(&vm->heap, haystack);
                vm_release(&vm->heap, needle);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_CONTAINS: not strings");
            }
            bool result = vmstring_contains(haystack.as.string, needle.as.string);
            vm_release(&vm->heap, haystack);
            vm_release(&vm->heap, needle);
            stack_push(vm, val_bool(result));
            break;
        }

        case OP_STR_EQ: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag != TAG_STRING || b.tag != TAG_STRING) {
                vm_release(&vm->heap, a);
                vm_release(&vm->heap, b);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_EQ: not strings");
            }
            bool result = vmstring_equal(a.as.string, b.as.string);
            vm_release(&vm->heap, a);
            vm_release(&vm->heap, b);
            stack_push(vm, val_bool(result));
            break;
        }

        case OP_STR_CHAR_AT: {
            NanoValue idx_v = stack_pop(vm);
            NanoValue s = stack_pop(vm);
            if (s.tag != TAG_STRING) {
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_CHAR_AT: not a string");
            }
            int64_t idx = (idx_v.tag == TAG_INT ? idx_v.as.i64 : 0);
            const char *str = vmstring_cstr(s.as.string);
            int64_t len = str ? (int64_t)strlen(str) : 0;
            int64_t ch = (idx >= 0 && idx < len) ? (unsigned char)str[idx] : -1;
            vm_release(&vm->heap, s);
            stack_push(vm, val_int(ch));
            break;
        }

        case OP_STR_FROM_INT: {
            NanoValue v = stack_pop(vm);
            VmString *s = vm_string_from_int(&vm->heap, v.tag == TAG_INT ? v.as.i64 : 0);
            stack_push(vm, val_string(s));
            break;
        }

        case OP_STR_FROM_FLOAT: {
            NanoValue v = stack_pop(vm);
            VmString *s = vm_string_from_float(&vm->heap, v.tag == TAG_FLOAT ? v.as.f64 : 0.0);
            stack_push(vm, val_string(s));
            break;
        }

        case OP_STR_TRIM: {
            NanoValue s = stack_pop(vm);
            if (s.tag != TAG_STRING) {
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_TRIM: not a string");
            }
            const char *str = vmstring_cstr(s.as.string);
            int64_t len = str ? (int64_t)strlen(str) : 0;
            int64_t start = 0;
            while (start < len && (str[start] == ' ' || str[start] == '\t' ||
                                   str[start] == '\n' || str[start] == '\r')) {
                start++;
            }
            int64_t end = len;
            while (end > start && (str[end - 1] == ' ' || str[end - 1] == '\t' ||
                                   str[end - 1] == '\n' || str[end - 1] == '\r')) {
                end--;
            }
            VmString *out = vm_string_new(&vm->heap, str + start, (uint32_t)(end - start));
            vm_release(&vm->heap, s);
            stack_push(vm, val_string(out));
            break;
        }

        case OP_STR_TO_LOWER:
        case OP_STR_TO_UPPER: {
            bool to_lower = (instr.opcode == OP_STR_TO_LOWER);
            NanoValue s = stack_pop(vm);
            if (s.tag != TAG_STRING) {
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  to_lower ? "STR_TO_LOWER: not a string"
                                           : "STR_TO_UPPER: not a string");
            }
            const char *str = vmstring_cstr(s.as.string);
            int64_t len = str ? (int64_t)strlen(str) : 0;
            /* Strings are interned/immutable, so transform into a scratch
             * buffer and only then construct the result string. */
            char stackbuf[256];
            char *buf = (len < (int64_t)sizeof(stackbuf)) ? stackbuf
                                                          : malloc((size_t)len + 1);
            if (!buf) {
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_MEMORY, "STR_TO_LOWER/UPPER: alloc failed");
            }
            for (int64_t i = 0; i < len; i++) {
                unsigned char c = (unsigned char)str[i];
                if (to_lower) {
                    buf[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
                } else {
                    buf[i] = (c >= 'a' && c <= 'z') ? (char)(c - 32) : (char)c;
                }
            }
            buf[len] = '\0';
            VmString *out = vm_string_new(&vm->heap, buf, (uint32_t)len);
            if (buf != stackbuf) free(buf);
            vm_release(&vm->heap, s);
            stack_push(vm, val_string(out));
            break;
        }

        case OP_STR_STARTS_WITH:
        case OP_STR_ENDS_WITH: {
            bool starts = (instr.opcode == OP_STR_STARTS_WITH);
            NanoValue affix_v = stack_pop(vm);
            NanoValue s = stack_pop(vm);
            if (s.tag != TAG_STRING || affix_v.tag != TAG_STRING) {
                vm_release(&vm->heap, affix_v);
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  starts ? "STR_STARTS_WITH: not a string"
                                         : "STR_ENDS_WITH: not a string");
            }
            const char *str = vmstring_cstr(s.as.string);
            const char *affix = vmstring_cstr(affix_v.as.string);
            size_t slen = str ? strlen(str) : 0;
            size_t alen = affix ? strlen(affix) : 0;
            bool result;
            if (alen > slen) {
                result = false;
            } else if (alen == 0) {
                result = true;
            } else if (starts) {
                result = memcmp(str, affix, alen) == 0;
            } else {
                result = memcmp(str + slen - alen, affix, alen) == 0;
            }
            vm_release(&vm->heap, affix_v);
            vm_release(&vm->heap, s);
            stack_push(vm, val_bool(result));
            break;
        }

        case OP_STR_SPLIT: {
            NanoValue delim_v = stack_pop(vm);
            NanoValue s = stack_pop(vm);
            if (s.tag != TAG_STRING || delim_v.tag != TAG_STRING) {
                vm_release(&vm->heap, delim_v);
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_SPLIT: not a string");
            }
            const char *str = vmstring_cstr(s.as.string);
            const char *delim = vmstring_cstr(delim_v.as.string);
            size_t dlen = delim ? strlen(delim) : 0;
            VmArray *arr = vm_array_new(&vm->heap, TAG_STRING, 8);
            if (dlen == 0) {
                /* Empty delimiter: split into individual characters. */
                size_t slen = str ? strlen(str) : 0;
                for (size_t i = 0; i < slen; i++) {
                    VmString *ch = vm_string_new(&vm->heap, str + i, 1);
                    vm_array_push(&vm->heap, arr, val_string(ch));
                    vm_release(&vm->heap, val_string(ch));
                }
            } else {
                const char *start = str ? str : "";
                const char *found;
                while ((found = strstr(start, delim)) != NULL) {
                    VmString *seg = vm_string_new(&vm->heap, start,
                                                  (uint32_t)(found - start));
                    vm_array_push(&vm->heap, arr, val_string(seg));
                    vm_release(&vm->heap, val_string(seg));
                    start = found + dlen;
                }
                VmString *rest = vm_string_new(&vm->heap, start,
                                               (uint32_t)strlen(start));
                vm_array_push(&vm->heap, arr, val_string(rest));
                vm_release(&vm->heap, val_string(rest));
            }
            vm_release(&vm->heap, delim_v);
            vm_release(&vm->heap, s);
            stack_push(vm, val_array(arr));
            break;
        }

        case OP_STR_REPLACE: {
            /* Args were pushed s, old, new -> pop in reverse. */
            NanoValue new_v = stack_pop(vm);
            NanoValue old_v = stack_pop(vm);
            NanoValue s = stack_pop(vm);
            if (s.tag != TAG_STRING || old_v.tag != TAG_STRING || new_v.tag != TAG_STRING) {
                vm_release(&vm->heap, new_v);
                vm_release(&vm->heap, old_v);
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STR_REPLACE: not a string");
            }
            const char *str = vmstring_cstr(s.as.string);
            const char *old_str = vmstring_cstr(old_v.as.string);
            const char *new_str = vmstring_cstr(new_v.as.string);
            size_t olen = old_str ? strlen(old_str) : 0;
            if (!str) str = "";
            if (olen == 0) {
                /* Empty needle: return the input unchanged (matches runtime). */
                VmString *copy = vm_string_new(&vm->heap, str, (uint32_t)strlen(str));
                vm_release(&vm->heap, new_v);
                vm_release(&vm->heap, old_v);
                vm_release(&vm->heap, s);
                stack_push(vm, val_string(copy));
                break;
            }
            size_t nlen = new_str ? strlen(new_str) : 0;
            /* Count occurrences to size the output buffer. */
            size_t count = 0;
            const char *p = str;
            const char *found;
            while ((found = strstr(p, old_str)) != NULL) { count++; p = found + olen; }
            size_t slen = strlen(str);
            /* out_len = slen + count*(nlen - olen); compute signed to be safe. */
            long long out_len_signed = (long long)slen +
                                       (long long)count * ((long long)nlen - (long long)olen);
            size_t out_len = out_len_signed > 0 ? (size_t)out_len_signed : 0;
            char stackbuf[512];
            char *buf = (out_len < sizeof(stackbuf)) ? stackbuf : malloc(out_len + 1);
            if (!buf) {
                vm_release(&vm->heap, new_v);
                vm_release(&vm->heap, old_v);
                vm_release(&vm->heap, s);
                return trap_error(vm, VM_ERR_MEMORY, "STR_REPLACE: alloc failed");
            }
            char *dst = buf;
            const char *src = str;
            while ((found = strstr(src, old_str)) != NULL) {
                size_t seg = (size_t)(found - src);
                memcpy(dst, src, seg); dst += seg;
                memcpy(dst, new_str, nlen); dst += nlen;
                src = found + olen;
            }
            size_t rest = strlen(src);
            memcpy(dst, src, rest); dst += rest;
            *dst = '\0';
            VmString *out = vm_string_new(&vm->heap, buf, (uint32_t)(dst - buf));
            if (buf != stackbuf) free(buf);
            vm_release(&vm->heap, new_v);
            vm_release(&vm->heap, old_v);
            vm_release(&vm->heap, s);
            stack_push(vm, val_string(out));
            break;
        }

        /* ============================================================
         * Array Ops
         * ============================================================ */

        case OP_ARR_NEW: {
            uint8_t elem_type = instr.operands[0].u8;
            VmArray *a = vm_array_new(&vm->heap, elem_type, 8);
            stack_push(vm, val_array(a));
            break;
        }

        case OP_ARR_PUSH: {
            NanoValue v = stack_pop(vm);
            NanoValue arr = stack_pop(vm);
            if (arr.tag != TAG_ARRAY) {
                vm_release(&vm->heap, arr);
                vm_release(&vm->heap, v);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "ARR_PUSH: not an array");
            }
            vm_array_push(&vm->heap, arr.as.array, v);
            vm_release(&vm->heap, v); /* push retains */
            stack_push(vm, arr);
            break;
        }

        case OP_ARR_POP: {
            NanoValue arr = stack_pop(vm);
            if (arr.tag != TAG_ARRAY) {
                vm_release(&vm->heap, arr);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "ARR_POP: not an array");
            }
            NanoValue v = vm_array_pop(arr.as.array);
            stack_push(vm, v);
            stack_push(vm, arr);
            break;
        }

        case OP_ARR_GET: {
            NanoValue idx_v = stack_pop(vm);
            NanoValue arr = stack_pop(vm);
            if (arr.tag != TAG_ARRAY) {
                vm_release(&vm->heap, arr);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "ARR_GET: not an array");
            }
            uint32_t idx = (uint32_t)(idx_v.tag == TAG_INT ? idx_v.as.i64 : 0);
            NanoValue v = vm_array_get(arr.as.array, idx);
            vm_retain(&vm->heap, v);
            vm_release(&vm->heap, arr);
            stack_push(vm, v);
            break;
        }

        case OP_ARR_SET: {
            NanoValue v = stack_pop(vm);
            NanoValue idx_v = stack_pop(vm);
            NanoValue arr = stack_pop(vm);
            if (arr.tag != TAG_ARRAY) {
                vm_release(&vm->heap, arr);
                vm_release(&vm->heap, v);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "ARR_SET: not an array");
            }
            uint32_t idx = (uint32_t)(idx_v.tag == TAG_INT ? idx_v.as.i64 : 0);
            vm_release(&vm->heap, vm_array_get(arr.as.array, idx));
            vm_array_set(arr.as.array, idx, v);
            stack_push(vm, arr);
            break;
        }

        case OP_ARR_LEN: {
            NanoValue arr = stack_pop(vm);
            if (arr.tag != TAG_ARRAY) {
                vm_release(&vm->heap, arr);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "ARR_LEN: not an array");
            }
            int64_t len = arr.as.array->length;
            vm_release(&vm->heap, arr);
            stack_push(vm, val_int(len));
            break;
        }

        case OP_ARR_SLICE: {
            NanoValue end_v = stack_pop(vm);
            NanoValue start_v = stack_pop(vm);
            NanoValue arr = stack_pop(vm);
            if (arr.tag != TAG_ARRAY) {
                vm_release(&vm->heap, arr);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "ARR_SLICE: not an array");
            }
            uint32_t start = (uint32_t)(start_v.tag == TAG_INT ? start_v.as.i64 : 0);
            uint32_t end = (uint32_t)(end_v.tag == TAG_INT ? end_v.as.i64 : arr.as.array->length);
            VmArray *result = vm_array_slice(&vm->heap, arr.as.array, start, end);
            vm_release(&vm->heap, arr);
            stack_push(vm, val_array(result));
            break;
        }

        case OP_ARR_REMOVE: {
            NanoValue idx_v = stack_pop(vm);
            NanoValue arr = stack_pop(vm);
            if (arr.tag != TAG_ARRAY) {
                vm_release(&vm->heap, arr);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "ARR_REMOVE: not an array");
            }
            uint32_t idx = (uint32_t)(idx_v.tag == TAG_INT ? idx_v.as.i64 : 0);
            vm_array_remove(arr.as.array, idx);
            stack_push(vm, arr);
            break;
        }

        case OP_ARR_LITERAL: {
            uint8_t elem_type = instr.operands[0].u8;
            uint16_t count = instr.operands[1].u16;
            VmArray *a = vm_array_new(&vm->heap, elem_type, count > 0 ? count : 8);
            /* Pop count values in reverse (they were pushed in order) */
            for (uint16_t i = 0; i < count; i++) {
                a->elements[count - 1 - i] = stack_pop(vm);
            }
            a->length = count;
            stack_push(vm, val_array(a));
            break;
        }

        /* ============================================================
         * Struct Ops
         * ============================================================ */

        case OP_STRUCT_NEW: {
            uint32_t def_idx = instr.operands[0].u32;
            VmStruct *s = vm_struct_new(&vm->heap, def_idx, 0);
            stack_push(vm, val_struct(s));
            break;
        }

        case OP_STRUCT_GET: {
            uint16_t field_idx = instr.operands[0].u16;
            NanoValue sv = stack_pop(vm);
            if (sv.tag != TAG_STRUCT || !sv.as.sval) {
                vm_release(&vm->heap, sv);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STRUCT_GET: not a struct");
            }
            if (field_idx >= sv.as.sval->field_count) {
                vm_release(&vm->heap, sv);
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS, "STRUCT_GET: field %u out of range", field_idx);
            }
            NanoValue v = sv.as.sval->fields[field_idx];
            vm_retain(&vm->heap, v);
            vm_release(&vm->heap, sv);
            stack_push(vm, v);
            break;
        }

        case OP_STRUCT_SET: {
            uint16_t field_idx = instr.operands[0].u16;
            NanoValue v = stack_pop(vm);
            NanoValue sv = stack_pop(vm);
            if (sv.tag != TAG_STRUCT || !sv.as.sval) {
                vm_release(&vm->heap, sv);
                vm_release(&vm->heap, v);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "STRUCT_SET: not a struct");
            }
            if (field_idx >= sv.as.sval->field_count) {
                vm_release(&vm->heap, sv);
                vm_release(&vm->heap, v);
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS, "STRUCT_SET: field %u out of range", field_idx);
            }
            vm_release(&vm->heap, sv.as.sval->fields[field_idx]);
            sv.as.sval->fields[field_idx] = v;
            stack_push(vm, sv);
            break;
        }

        case OP_STRUCT_LITERAL: {
            uint32_t def_idx = instr.operands[0].u32;
            uint16_t field_count = instr.operands[1].u16;
            VmStruct *s = vm_struct_new(&vm->heap, def_idx, field_count);
            /* Pop fields in reverse order */
            for (uint16_t i = 0; i < field_count; i++) {
                s->fields[field_count - 1 - i] = stack_pop(vm);
            }
            stack_push(vm, val_struct(s));
            break;
        }

        /* ============================================================
         * Union/Enum Ops
         * ============================================================ */

        case OP_UNION_CONSTRUCT: {
            uint32_t def_idx = instr.operands[0].u32;
            uint16_t variant = instr.operands[1].u16;
            uint16_t fcount = instr.operands[2].u16;
            VmUnion *u = vm_union_new(&vm->heap, def_idx, variant, fcount);
            for (uint16_t i = 0; i < fcount; i++) {
                u->fields[fcount - 1 - i] = stack_pop(vm);
            }
            stack_push(vm, val_union(u));
            break;
        }

        case OP_UNION_TAG: {
            NanoValue uv = stack_pop(vm);
            if (uv.tag != TAG_UNION || !uv.as.uval) {
                vm_release(&vm->heap, uv);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "UNION_TAG: not a union");
            }
            int64_t tag = uv.as.uval->variant;
            vm_release(&vm->heap, uv);
            stack_push(vm, val_int(tag));
            break;
        }

        case OP_UNION_FIELD: {
            uint16_t field_idx = instr.operands[0].u16;
            NanoValue uv = stack_pop(vm);
            if (uv.tag != TAG_UNION || !uv.as.uval) {
                vm_release(&vm->heap, uv);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "UNION_FIELD: not a union");
            }
            if (field_idx >= uv.as.uval->field_count) {
                vm_release(&vm->heap, uv);
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS, "UNION_FIELD: field %u out of range", field_idx);
            }
            NanoValue v = uv.as.uval->fields[field_idx];
            vm_retain(&vm->heap, v);
            vm_release(&vm->heap, uv);
            stack_push(vm, v);
            break;
        }

        case OP_MATCH_TAG: {
            uint16_t variant = instr.operands[0].u16;
            int32_t offset = instr.operands[1].i32;
            NanoValue top = stack_peek(vm, 0);
            if (top.tag == TAG_UNION && top.as.uval && top.as.uval->variant == variant) {
                /* Match - jump to arm */
                vm->ip = (uint32_t)((int32_t)instr_start + offset);
            }
            /* No match - fall through to next MATCH_TAG */
            break;
        }

        case OP_ENUM_VAL: {
            uint32_t def_idx = instr.operands[0].u32;
            uint16_t variant = instr.operands[1].u16;
            (void)def_idx;
            stack_push(vm, val_enum((int32_t)variant));
            break;
        }

        /* ============================================================
         * Tuple Ops
         * ============================================================ */

        case OP_TUPLE_NEW: {
            uint16_t count = instr.operands[0].u16;
            VmTuple *t = vm_tuple_new(&vm->heap, count);
            for (uint16_t i = 0; i < count; i++) {
                t->elements[count - 1 - i] = stack_pop(vm);
            }
            stack_push(vm, val_tuple(t));
            break;
        }

        case OP_TUPLE_GET: {
            uint16_t index = instr.operands[0].u16;
            NanoValue tv = stack_pop(vm);
            if (tv.tag != TAG_TUPLE || !tv.as.tuple) {
                vm_release(&vm->heap, tv);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "TUPLE_GET: not a tuple");
            }
            if (index >= tv.as.tuple->count) {
                vm_release(&vm->heap, tv);
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS, "TUPLE_GET: index %u out of range", index);
            }
            NanoValue v = tv.as.tuple->elements[index];
            vm_retain(&vm->heap, v);
            vm_release(&vm->heap, tv);
            stack_push(vm, v);
            break;
        }

        /* ============================================================
         * Hashmap Ops
         * ============================================================ */

        case OP_HM_NEW: {
            uint8_t key_type = instr.operands[0].u8;
            uint8_t val_type = instr.operands[1].u8;
            VmHashMap *m = vm_hashmap_new(&vm->heap, key_type, val_type);
            stack_push(vm, val_hashmap(m));
            break;
        }

        case OP_HM_GET: {
            NanoValue key = stack_pop(vm);
            NanoValue map = stack_pop(vm);
            if (map.tag != TAG_HASHMAP) {
                vm_release(&vm->heap, map);
                vm_release(&vm->heap, key);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "HM_GET: not a hashmap");
            }
            NanoValue v = vm_hashmap_get(map.as.hashmap, key);
            vm_retain(&vm->heap, v);
            vm_release(&vm->heap, map);
            vm_release(&vm->heap, key);
            stack_push(vm, v);
            break;
        }

        case OP_HM_SET: {
            NanoValue v = stack_pop(vm);
            NanoValue key = stack_pop(vm);
            NanoValue map = stack_pop(vm);
            if (map.tag != TAG_HASHMAP) {
                vm_release(&vm->heap, map);
                vm_release(&vm->heap, key);
                vm_release(&vm->heap, v);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "HM_SET: not a hashmap");
            }
            vm_hashmap_set(&vm->heap, map.as.hashmap, key, v);
            vm_release(&vm->heap, key);
            vm_release(&vm->heap, v);
            stack_push(vm, map);
            break;
        }

        case OP_HM_HAS: {
            NanoValue key = stack_pop(vm);
            NanoValue map = stack_pop(vm);
            if (map.tag != TAG_HASHMAP) {
                vm_release(&vm->heap, map);
                vm_release(&vm->heap, key);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "HM_HAS: not a hashmap");
            }
            bool has = vm_hashmap_has(map.as.hashmap, key);
            vm_release(&vm->heap, map);
            vm_release(&vm->heap, key);
            stack_push(vm, val_bool(has));
            break;
        }

        case OP_HM_DELETE: {
            NanoValue key = stack_pop(vm);
            NanoValue map = stack_pop(vm);
            if (map.tag != TAG_HASHMAP) {
                vm_release(&vm->heap, map);
                vm_release(&vm->heap, key);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "HM_DELETE: not a hashmap");
            }
            vm_hashmap_delete(&vm->heap, map.as.hashmap, key);
            vm_release(&vm->heap, key);
            stack_push(vm, map);
            break;
        }

        case OP_HM_KEYS: {
            NanoValue map = stack_pop(vm);
            if (map.tag != TAG_HASHMAP) {
                vm_release(&vm->heap, map);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "HM_KEYS: not a hashmap");
            }
            VmArray *keys = vm_hashmap_keys(&vm->heap, map.as.hashmap);
            vm_release(&vm->heap, map);
            stack_push(vm, val_array(keys));
            break;
        }

        case OP_HM_VALUES: {
            NanoValue map = stack_pop(vm);
            if (map.tag != TAG_HASHMAP) {
                vm_release(&vm->heap, map);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "HM_VALUES: not a hashmap");
            }
            VmArray *vals = vm_hashmap_values(&vm->heap, map.as.hashmap);
            vm_release(&vm->heap, map);
            stack_push(vm, val_array(vals));
            break;
        }

        case OP_HM_LEN: {
            NanoValue map = stack_pop(vm);
            if (map.tag != TAG_HASHMAP) {
                vm_release(&vm->heap, map);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "HM_LEN: not a hashmap");
            }
            int64_t len = map.as.hashmap->count;
            vm_release(&vm->heap, map);
            stack_push(vm, val_int(len));
            break;
        }

        /* ============================================================
         * GC/Memory
         * ============================================================ */

        case OP_GC_RETAIN: {
            NanoValue v = stack_peek(vm, 0);
            vm_retain(&vm->heap, v);
            break;
        }

        case OP_GC_RELEASE: {
            NanoValue v = stack_pop(vm);
            vm_release(&vm->heap, v);
            break;
        }

        case OP_GC_SCOPE_ENTER:
            /* Scope tracking is implicit in the call stack */
            break;

        case OP_GC_SCOPE_EXIT:
            /* Scope tracking is implicit in the call stack */
            break;

        /* ============================================================
         * Type Casts
         * ============================================================ */

        case OP_CAST_INT: {
            NanoValue v = stack_pop(vm);
            switch (v.tag) {
                case TAG_INT:   stack_push(vm, v); break;
                case TAG_FLOAT: stack_push(vm, val_int((int64_t)v.as.f64)); break;
                case TAG_BOOL:  stack_push(vm, val_int(v.as.boolean ? 1 : 0)); break;
                case TAG_U8:    stack_push(vm, val_int(v.as.u8)); break;
                case TAG_ENUM:  stack_push(vm, val_int(v.as.i64)); break;
                case TAG_STRING: {
                    const char *str = vmstring_cstr(v.as.string);
                    int64_t result = str ? strtoll(str, NULL, 10) : 0;
                    vm_release(&vm->heap, v);
                    stack_push(vm, val_int(result));
                    break;
                }
                default:
                    vm_release(&vm->heap, v);
                    stack_push(vm, val_int(0));
                    break;
            }
            break;
        }

        case OP_CAST_FLOAT: {
            NanoValue v = stack_pop(vm);
            switch (v.tag) {
                case TAG_FLOAT: stack_push(vm, v); break;
                case TAG_INT:   stack_push(vm, val_float((double)v.as.i64)); break;
                case TAG_BOOL:  stack_push(vm, val_float(v.as.boolean ? 1.0 : 0.0)); break;
                case TAG_STRING: {
                    const char *str = vmstring_cstr(v.as.string);
                    double result = str ? strtod(str, NULL) : 0.0;
                    vm_release(&vm->heap, v);
                    stack_push(vm, val_float(result));
                    break;
                }
                default:
                    vm_release(&vm->heap, v);
                    stack_push(vm, val_float(0.0));
                    break;
            }
            break;
        }

        case OP_CAST_BOOL: {
            NanoValue v = stack_pop(vm);
            bool result = val_truthy(v);
            vm_release(&vm->heap, v);
            stack_push(vm, val_bool(result));
            break;
        }

        case OP_CAST_STRING: {
            NanoValue v = stack_pop(vm);
            VmString *s;
            switch (v.tag) {
                case TAG_STRING: stack_push(vm, v); break; /* already a string */
                case TAG_INT:
                    s = vm_string_from_int(&vm->heap, v.as.i64);
                    stack_push(vm, val_string(s));
                    break;
                case TAG_FLOAT:
                    s = vm_string_from_float(&vm->heap, v.as.f64);
                    stack_push(vm, val_string(s));
                    break;
                case TAG_BOOL:
                    s = vm_string_from_bool(&vm->heap, v.as.boolean);
                    stack_push(vm, val_string(s));
                    break;
                default:
                    vm_release(&vm->heap, v);
                    s = vm_string_new(&vm->heap, "", 0);
                    stack_push(vm, val_string(s));
                    break;
            }
            break;
        }

        case OP_TYPE_CHECK: {
            uint8_t expected = instr.operands[0].u8;
            NanoValue v = stack_pop(vm);
            stack_push(vm, val_bool(v.tag == expected));
            vm_release(&vm->heap, v);
            break;
        }

        /* ============================================================
         * Closures
         * ============================================================ */

        case OP_CLOSURE_NEW: {
            uint32_t fn_idx_c = instr.operands[0].u32;
            uint16_t capture_count = instr.operands[1].u16;
            VmClosure *c = vm_closure_new(&vm->heap, fn_idx_c, capture_count);
            /* Pop captures from stack (pushed in order, stored in order) */
            for (int16_t i = (int16_t)(capture_count - 1); i >= 0; i--) {
                c->captures[i] = stack_pop(vm);
            }
            stack_push(vm, val_closure(c));
            break;
        }

        case OP_CLOSURE_CALL: {
            NanoValue fn_val = stack_pop(vm);
            if (fn_val.tag != TAG_CLOSURE || !fn_val.as.closure) {
                vm_release(&vm->heap, fn_val);
                return trap_error(vm, VM_ERR_TYPE_ERROR, "CLOSURE_CALL: not a closure");
            }
            VmClosure *closure = fn_val.as.closure;
            uint32_t callee_idx = closure->fn_idx;
            if (callee_idx >= vm->module->function_count) {
                return trap_error(vm, VM_ERR_UNDEFINED_FUNCTION, "Closure fn %u not found", callee_idx);
            }
            const NvmFunctionEntry *callee = &vm->module->functions[callee_idx];
            if (vm->frame_count >= VM_MAX_FRAMES) {
                return trap_error(vm, VM_ERR_CALL_DEPTH, "Call depth exceeded");
            }

            uint32_t new_base = vm->stack_size - callee->arity;
            for (uint16_t i = callee->arity; i < callee->local_count; i++) {
                stack_push(vm, val_void());
            }

            VmCallFrame *new_frame = &vm->frames[vm->frame_count++];
            new_frame->fn_idx = callee_idx;
            new_frame->return_ip = vm->ip;
            new_frame->stack_base = new_base;
            new_frame->local_count = callee->local_count;
            new_frame->closure = closure;
            new_frame->module = vm->module;
            new_frame->current_line = 0;
            new_frame->current_col  = 0;

            frame = new_frame;
            vm->current_fn = callee_idx;
            vm->ip = callee->code_offset;
            code_end = callee->code_offset + callee->code_length;
            break;
        }

        /* ============================================================
         * I/O & Debug
         * ============================================================ */

        case OP_PRINT: {
            if (vm->profile.enabled) vm->profile.traps++;
            VmTrap t = { .type = TRAP_PRINT };
            t.data.print.value = stack_pop(vm);
            t.data.print.newline = false;
            return t;
        }

        case OP_PRINTLN: {
            if (vm->profile.enabled) vm->profile.traps++;
            VmTrap t = { .type = TRAP_PRINT };
            t.data.print.value = stack_pop(vm);
            t.data.print.newline = true;
            return t;
        }

        case OP_ASSERT: {
            if (vm->profile.enabled) vm->profile.traps++;
            VmTrap t = { .type = TRAP_ASSERT };
            t.data.assert_check.condition = stack_pop(vm);
            return t;
        }

        case OP_DEBUG_LINE:
            /* Track source line in the current frame for stack traces.
             * Column is resolved via debug_entries; reset to 0 here. */
            frame->current_line = instr.operands[0].u32;
            frame->current_col  = 0;
            /* Resolve column from debug_entries if available */
            {
                const NvmModule *dbg_mod = vm->module;
                if (dbg_mod && dbg_mod->debug_count > 0) {
                    uint32_t best_off = 0;
                    bool col_found = false;
                    for (uint32_t d = 0; d < dbg_mod->debug_count; d++) {
                        uint32_t off = dbg_mod->debug_entries[d].bytecode_offset;
                        if (off <= instr_start && (!col_found || off >= best_off)) {
                            best_off = off;
                            if (dbg_mod->debug_entries[d].source_line == frame->current_line)
                                frame->current_col = dbg_mod->debug_entries[d].source_col;
                            col_found = true;
                        }
                    }
                }
            }
            break;

        case OP_HALT:
            if (vm->profile.enabled) vm->profile.traps++;
            return trap_halt();

        /* ============================================================
         * Opaque Proxy
         * ============================================================ */

        case OP_OPAQUE_NULL: {
            NanoValue v = {0};
            v.tag = TAG_OPAQUE;
            v.as.proxy_id = 0;
            stack_push(vm, v);
            break;
        }

        case OP_OPAQUE_VALID: {
            NanoValue v = stack_pop(vm);
            stack_push(vm, val_bool(v.tag == TAG_OPAQUE && v.as.proxy_id != 0));
            break;
        }

        case OP_MEM_LOAD8:
        case OP_MEM_LOAD16:
        case OP_MEM_LOAD32:
        case OP_MEM_LOAD64: {
            NanoValue address = stack_pop(vm);
            if (address.tag != TAG_INT || address.as.i64 < 0)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires a non-negative integer address",
                                  isa_get_info(instr.opcode)->name);
            uint64_t width = instr.opcode == OP_MEM_LOAD8 ? 1
                : instr.opcode == OP_MEM_LOAD16 ? 2
                : instr.opcode == OP_MEM_LOAD32 ? 4 : 8;
            uint64_t start = (uint64_t)address.as.i64;
            if (start > vm->memory_size || width > vm->memory_size - start)
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS,
                                  "Memory load at %llu exceeds %llu bytes",
                                  (unsigned long long)start,
                                  (unsigned long long)vm->memory_size);
            uint64_t value = 0;
            for (uint64_t i = 0; i < width; i++)
                value |= (uint64_t)vm->memory[start + i] << (i * 8);
            stack_push(vm, val_int((int64_t)value));
            break;
        }

        case OP_MEM_STORE8:
        case OP_MEM_STORE16:
        case OP_MEM_STORE32:
        case OP_MEM_STORE64: {
            NanoValue value = stack_pop(vm);
            NanoValue address = stack_pop(vm);
            if (address.tag != TAG_INT || value.tag != TAG_INT || address.as.i64 < 0)
                return trap_error(vm, VM_ERR_TYPE_ERROR,
                                  "%s requires integer address and value",
                                  isa_get_info(instr.opcode)->name);
            uint64_t width = instr.opcode == OP_MEM_STORE8 ? 1
                : instr.opcode == OP_MEM_STORE16 ? 2
                : instr.opcode == OP_MEM_STORE32 ? 4 : 8;
            uint64_t start = (uint64_t)address.as.i64;
            if (start > vm->memory_size || width > vm->memory_size - start)
                return trap_error(vm, VM_ERR_OUT_OF_BOUNDS,
                                  "Memory store at %llu exceeds %llu bytes",
                                  (unsigned long long)start,
                                  (unsigned long long)vm->memory_size);
            uint64_t bits = (uint64_t)value.as.i64;
            for (uint64_t i = 0; i < width; i++)
                vm->memory[start + i] = (uint8_t)(bits >> (i * 8));
            break;
        }

        default:
            return trap_error(vm, VM_ERR_INVALID_OPCODE, "Unknown opcode 0x%02x", instr.opcode);

        } /* switch */
    } /* while */

    /* Fell off the end of function code without RET or HALT */
    /* Treat as implicit RET with void */
    if (vm->frame_count > 0) {
        NanoValue result = val_void();
        if (vm->stack_size > frame->stack_base + frame->local_count) {
            result = stack_pop(vm);
        }
        while (vm->stack_size > frame->stack_base) {
            NanoValue v = stack_pop(vm);
            vm_release(&vm->heap, v);
        }
        vm->frame_count--;
        if (vm->frame_count == 0) {
            stack_push(vm, result);
            return trap_none();
        }
        frame = &vm->frames[vm->frame_count - 1];
        vm->current_fn = frame->fn_idx;
        vm->ip = frame->return_ip;
        stack_push(vm, result);
    }

    return trap_none();
}

/* ========================================================================
 * Debug: Source-Mapped Stack Trace
 * ======================================================================== */

void vm_stack_trace(const VmState *vm, FILE *out) {
    if (!out) out = stderr;
    fprintf(out, "Stack trace (most recent call first):\n");
    if (vm->frame_count == 0) {
        fprintf(out, "  (no frames)\n");
        return;
    }
    for (int i = (int)vm->frame_count - 1; i >= 0; i--) {
        const VmCallFrame *frame = &vm->frames[i];
        const NvmModule *mod = frame->module ? frame->module : vm->module;

        /* Resolve function name */
        const char *fn_name = "??";
        if (mod && frame->fn_idx < mod->function_count) {
            const char *s = nvm_get_string(mod, mod->functions[frame->fn_idx].name_idx);
            if (s && s[0]) fn_name = s;
        }

        /* Resolve source file: prefer module's source_file_idx, fall back to string 0 */
        const char *src_file = "<unknown>";
        if (mod) {
            if (mod->source_file_idx > 0) {
                const char *sf = nvm_get_string(mod, mod->source_file_idx);
                if (sf && sf[0]) src_file = sf;
            } else {
                /* Legacy: use string 0 as module/file name */
                const char *s0 = nvm_get_string(mod, 0);
                if (s0 && s0[0]) src_file = s0;
            }
        }

        /* Source line and col: prefer per-frame tracking, fall back to debug entries */
        uint32_t line = frame->current_line;
        uint32_t col  = frame->current_col;
        if (line == 0 && mod && mod->debug_count > 0) {
            /* Find the debug entry with the largest bytecode_offset <= frame's ip.
             * For frames other than the top frame we don't have a saved ip,
             * so we use frame->return_ip as a proxy. */
            uint32_t search_ip = (i == (int)vm->frame_count - 1)
                                  ? vm->ip
                                  : frame->return_ip;
            uint32_t best_line = 0;
            uint32_t best_col  = 0;
            uint32_t best_offset = 0;
            bool found = false;
            for (uint32_t d = 0; d < mod->debug_count; d++) {
                uint32_t off = mod->debug_entries[d].bytecode_offset;
                if (off <= search_ip) {
                    if (!found || off >= best_offset) {
                        best_offset = off;
                        best_line   = mod->debug_entries[d].source_line;
                        best_col    = mod->debug_entries[d].source_col;
                        found = true;
                    }
                }
            }
            if (found) { line = best_line; col = best_col; }
        }

        int frame_num = (int)vm->frame_count - 1 - i;
        if (line > 0 && col > 0) {
            fprintf(out, "  #%-2d  %s  %s:%u:%u\n",
                    frame_num, fn_name, src_file, line, col);
        } else if (line > 0) {
            fprintf(out, "  #%-2d  %s  %s:%u\n",
                    frame_num, fn_name, src_file, line);
        } else {
            fprintf(out, "  #%-2d  %s  %s:?\n",
                    frame_num, fn_name, src_file);
        }
    }
}

/* ========================================================================
 * Runtime Harness (the "co-processor")
 *
 * Calls vm_core_execute() in a loop, handling each trap that the
 * NanoISA core returns.  In the software VM both layers run in the
 * same process.  On an FPGA the harness would run on the host CPU
 * and communicate with the core over PCIe/AXI.
 * ======================================================================== */

VmResult vm_call_function(VmState *vm, uint32_t fn_idx, NanoValue *args, uint16_t arg_count) {
    if (fn_idx >= vm->module->function_count) {
        return vm_error(vm, VM_ERR_UNDEFINED_FUNCTION, "Function %u out of range", fn_idx);
    }

    const NvmFunctionEntry *fn = &vm->module->functions[fn_idx];

    /* Push a call frame */
    if (vm->frame_count >= VM_MAX_FRAMES) {
        return vm_error(vm, VM_ERR_CALL_DEPTH, "Call depth exceeded");
    }

    uint32_t stack_base = vm->stack_size;

    /* Push args as first locals */
    for (uint16_t i = 0; i < arg_count; i++) {
        VmResult r = stack_push(vm, args[i]);
        if (r != VM_OK) return r;
    }

    /* Push remaining locals as void */
    for (uint16_t i = arg_count; i < fn->local_count; i++) {
        VmResult r = stack_push(vm, val_void());
        if (r != VM_OK) return r;
    }

    VmCallFrame *frame = &vm->frames[vm->frame_count++];
    frame->fn_idx = fn_idx;
    frame->return_ip = vm->ip;
    frame->stack_base = stack_base;
    frame->local_count = fn->local_count;
    frame->closure = NULL;
    frame->module = vm->module;
    frame->current_line = 0;
    frame->current_col  = 0;

    vm->current_fn = fn_idx;
    vm->ip = fn->code_offset;

    /* Run the core in a loop, handling traps */
    for (;;) {
        VmTrap trap = vm_core_execute(vm);

        switch (trap.type) {
        case TRAP_NONE:
            return VM_OK;

        case TRAP_HALT:
            return VM_OK;

        case TRAP_PRINT:
            val_print(trap.data.print.value, vm_out(vm));
            if (trap.data.print.newline) fprintf(vm_out(vm), "\n");
            vm_release(&vm->heap, trap.data.print.value);
            break;

        case TRAP_ASSERT:
            if (!val_truthy(trap.data.assert_check.condition)) {
                vm_release(&vm->heap, trap.data.assert_check.condition);
                return vm_error(vm, VM_ERR_ASSERT_FAILED, "Assertion failed");
            }
            vm_release(&vm->heap, trap.data.assert_check.condition);
            break;

        case TRAP_EXTERN_CALL: {
            NanoValue ext_result;
            char ext_err[256];
            bool ffi_ok;
            struct timespec ffi_start = {0}, ffi_stop = {0};
            if (vm->profile.enabled) {
                clock_gettime(CLOCK_MONOTONIC, &ffi_start);
                uint8_t scratch[COP_MAILBOX_SLOT_SIZE];
                for (int i = 0; i < trap.data.extern_call.argc; i++) {
                    vm->profile.ffi_request_bytes += cop_serialize_value(
                        &trap.data.extern_call.args[i], scratch, sizeof(scratch));
                }
            }
            if (vm->isolate_ffi) {
                ffi_ok = vm_ffi_call_cop(vm, vm->module, trap.data.extern_call.import_idx,
                                         trap.data.extern_call.args, trap.data.extern_call.argc,
                                         &ext_result, &vm->heap,
                                         ext_err, sizeof(ext_err));
            } else {
                ffi_ok = vm_ffi_call(vm->module, trap.data.extern_call.import_idx,
                                     trap.data.extern_call.args, trap.data.extern_call.argc,
                                     &ext_result, &vm->heap,
                                     ext_err, sizeof(ext_err));
            }
            if (vm->profile.enabled) {
                clock_gettime(CLOCK_MONOTONIC, &ffi_stop);
                int64_t seconds = ffi_stop.tv_sec - ffi_start.tv_sec;
                int64_t nanoseconds = ffi_stop.tv_nsec - ffi_start.tv_nsec;
                vm->profile.ffi_elapsed_ns +=
                    (uint64_t)(seconds * 1000000000LL + nanoseconds);
                if (!ffi_ok) {
                    vm->profile.ffi_failures++;
                } else {
                    uint8_t scratch[COP_MAILBOX_SLOT_SIZE];
                    vm->profile.ffi_response_bytes += cop_serialize_value(
                        &ext_result, scratch, sizeof(scratch));
                }
            }
            if (!ffi_ok) {
                return vm_error(vm, VM_ERR_NOT_IMPLEMENTED,
                                "FFI call failed: %s", ext_err);
            }
            /* Push result back onto the VM stack for the core to consume */
            stack_push(vm, ext_result);
            break;
        }

        case TRAP_ERROR:
            /* Emit source-mapped stack trace before unwinding.
             * Always emit when debug_mode is on; also emit when the
             * binary was compiled with debug info. */
            if (vm->debug_mode || (vm->module->header.flags & NVM_FLAG_DEBUG_INFO)) {
                FILE *trace_out = vm->output ? vm->output : stderr;
                fprintf(trace_out, "\nRuntime error: %s\n",
                        vm_error_string(trap.data.error.code));
                if (vm->error_msg[0]) {
                    fprintf(trace_out, "  %s\n", vm->error_msg);
                }
                vm_stack_trace(vm, trace_out);
            }
            return trap.data.error.code;
        }
    }
}

VmResult vm_invoke(VmState *vm, uint32_t fn_idx, const NanoValue *args,
                   uint16_t arg_count, NanoValue *out_result) {
    if (!vm || !vm->module) return VM_ERR_UNDEFINED_FUNCTION;
    if (out_result) *out_result = val_void();
    if (vm->frame_count != 0) {
        return vm_error(vm, VM_ERR_CALL_DEPTH,
                        "Cannot invoke a function while the VM is executing");
    }
    if (fn_idx >= vm->module->function_count) {
        return vm_error(vm, VM_ERR_UNDEFINED_FUNCTION,
                        "Function %u out of range", fn_idx);
    }

    const NvmFunctionEntry *fn = &vm->module->functions[fn_idx];
    if (arg_count != fn->arity) {
        return vm_error(vm, VM_ERR_TYPE_ERROR,
                        "Function %u expects %u arguments, got %u",
                        fn_idx, fn->arity, arg_count);
    }
    if (arg_count > 0 && !args) {
        return vm_error(vm, VM_ERR_TYPE_ERROR,
                        "Function %u arguments are NULL", fn_idx);
    }

    uint32_t stack_base = vm->stack_size;
    uint32_t saved_ip = vm->ip;
    uint32_t saved_fn = vm->current_fn;
    const NvmModule *saved_module = vm->module;

    uint32_t required = stack_base + fn->local_count;
    if (required > vm->stack_capacity) {
        uint32_t new_capacity = vm->stack_capacity;
        while (new_capacity < required) new_capacity *= 2;
        NanoValue *new_stack = realloc(vm->stack,
                                       new_capacity * sizeof(NanoValue));
        if (!new_stack) {
            return vm_error(vm, VM_ERR_MEMORY, "Stack grow failed");
        }
        vm->stack = new_stack;
        vm->stack_capacity = new_capacity;
    }

    /* vm_call_function consumes argument ownership through its frame cleanup. */
    for (uint16_t i = 0; i < arg_count; i++) vm_retain(&vm->heap, args[i]);

    VmResult result = vm_call_function(vm, fn_idx, (NanoValue *)args, arg_count);
    NanoValue returned = val_void();
    if (result == VM_OK && vm->stack_size > stack_base) {
        returned = stack_pop(vm);
    }

    while (vm->stack_size > stack_base) {
        vm_release(&vm->heap, stack_pop(vm));
    }
    vm->frame_count = 0;
    vm->ip = saved_ip;
    vm->current_fn = saved_fn;
    vm->module = saved_module;

    if (result == VM_OK) {
        vm->last_error = VM_OK;
        vm->error_msg[0] = '\0';
        if (out_result) {
            *out_result = returned;
        } else {
            vm_release(&vm->heap, returned);
        }
    }
    return result;
}

VmResult vm_execute(VmState *vm) {
    if (!(vm->module->header.flags & NVM_FLAG_HAS_MAIN)) {
        return vm_error(vm, VM_ERR_UNDEFINED_FUNCTION, "No entry point defined");
    }

    uint32_t entry = vm->module->header.entry_point;
    if (entry >= vm->module->function_count) {
        return vm_error(vm, VM_ERR_UNDEFINED_FUNCTION, "Entry point %u out of range", entry);
    }

    /* Call __init__ to initialize globals before the entry point */
    for (uint32_t i = 0; i < vm->module->function_count; i++) {
        const char *fn_name = nvm_get_string(vm->module,
                                              vm->module->functions[i].name_idx);
        if (fn_name && strcmp(fn_name, "__init__") == 0) {
            VmResult ir = vm_call_function(vm, i, NULL, 0);
            if (ir != VM_OK) return ir;
            break;
        }
    }

    return vm_call_function(vm, entry, NULL, 0);
}
