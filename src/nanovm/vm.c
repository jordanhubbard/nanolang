/*
 * NanoVM - Bytecode execution engine
 *
 * Simple switch dispatch over all NanoISA opcodes.
 */

#include "vm.h"
#include "vm_ffi.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

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
    vm_heap_destroy(&vm->heap);
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

        case OP_DUP: {
            NanoValue top = stack_peek(vm, 0);
            vm_retain(top);
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
            vm_retain(v);
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
            vm_retain(v);
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
            uint16_t depth = instr.operands[0].u16;
            uint16_t idx = instr.operands[1].u16;
            (void)depth;
            /* Find the closure in the current frame */
            if (frame->closure && idx < frame->closure->capture_count) {
                NanoValue v = frame->closure->captures[idx];
                vm_retain(v);
                stack_push(vm, v);
            } else {
                stack_push(vm, val_void());
            }
            break;
        }

        case OP_STORE_UPVALUE: {
            uint16_t depth = instr.operands[0].u16;
            uint16_t idx = instr.operands[1].u16;
            (void)depth;
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

        case OP_ADD: {
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
                    vm_array_push(result, ev);
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
                    vm_array_push(result, ev);
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

        case OP_SUB: {
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
                    vm_array_push(result, ev);
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
                    vm_array_push(result, ev);
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

        case OP_MUL: {
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
                    vm_array_push(result, ev);
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
                    vm_array_push(result, ev);
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

        case OP_DIV: {
            NanoValue b = stack_pop(vm);
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_ENUM) { a = val_int((int64_t)a.as.enum_val); }
            if (b.tag == TAG_ENUM) { b = val_int((int64_t)b.as.enum_val); }
            if (a.tag == TAG_INT && b.tag == TAG_INT) {
                /* Division by zero = 0 (matches Coq semantics) */
                stack_push(vm, val_int(b.as.i64 == 0 ? 0 : a.as.i64 / b.as.i64));
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
                    vm_array_push(result, ev);
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
                    vm_array_push(result, ev);
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
                stack_push(vm, val_int(b.as.i64 == 0 ? 0 : a.as.i64 % b.as.i64));
            } else {
                return trap_error(vm, VM_ERR_TYPE_ERROR, "MOD: type error");
            }
            break;
        }

        case OP_NEG: {
            NanoValue a = stack_pop(vm);
            if (a.tag == TAG_INT) {
                stack_push(vm, val_int(-a.as.i64));
            } else if (a.tag == TAG_FLOAT) {
                stack_push(vm, val_float(-a.as.f64));
            } else {
                return trap_error(vm, VM_ERR_TYPE_ERROR, "NEG: type error");
            }
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

        /* ============================================================
         * Control Flow
         * ============================================================ */

        case OP_JMP: {
            int32_t offset = instr.operands[0].i32;
            vm->ip = (uint32_t)((int32_t)instr_start + offset);
            break;
        }

        case OP_JMP_TRUE: {
            NanoValue cond = stack_pop(vm);
            if (val_truthy(cond)) {
                vm->ip = (uint32_t)((int32_t)instr_start + instr.operands[0].i32);
            }
            vm_release(&vm->heap, cond);
            break;
        }

        case OP_JMP_FALSE: {
            NanoValue cond = stack_pop(vm);
            if (!val_truthy(cond)) {
                vm->ip = (uint32_t)((int32_t)instr_start + instr.operands[0].i32);
            }
            vm_release(&vm->heap, cond);
            break;
        }

        case OP_CALL: {
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

            /* Save current execution context */
            frame = new_frame;
            vm->current_fn = callee_idx;
            vm->ip = callee->code_offset;
            code_end = callee->code_offset + callee->code_length;
            break;
        }

        case OP_CALL_INDIRECT: {
            NanoValue fn_val = stack_pop(vm);
            if (fn_val.tag == TAG_FUNCTION) {
                /* Check if it's a closure */
                VmClosure *closure = NULL;
                uint32_t callee_idx;

                if (fn_val.as.closure &&
                    ((VmHeapHeader *)fn_val.as.closure)->obj_type == TAG_FUNCTION) {
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
            vm_array_push(arr.as.array, v);
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
            vm_retain(v);
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
            vm_retain(v);
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
            vm_retain(v);
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
            vm_retain(v);
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
            vm_retain(v);
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
            vm_retain(v);
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
            if (fn_val.tag != TAG_FUNCTION || !fn_val.as.closure) {
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
            VmTrap t = { .type = TRAP_PRINT };
            t.data.print.value = stack_pop(vm);
            return t;
        }

        case OP_ASSERT: {
            VmTrap t = { .type = TRAP_ASSERT };
            t.data.assert_check.condition = stack_pop(vm);
            return t;
        }

        case OP_DEBUG_LINE:
            /* Source line tracking - ignored during execution */
            break;

        case OP_HALT:
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
            fprintf(vm_out(vm), "\n");
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
            if (vm->isolate_ffi) {
                ffi_ok = vm_ffi_call_cop(vm->module, trap.data.extern_call.import_idx,
                                         trap.data.extern_call.args, trap.data.extern_call.argc,
                                         &ext_result, &vm->heap,
                                         ext_err, sizeof(ext_err));
            } else {
                ffi_ok = vm_ffi_call(vm->module, trap.data.extern_call.import_idx,
                                     trap.data.extern_call.args, trap.data.extern_call.argc,
                                     &ext_result, &vm->heap,
                                     ext_err, sizeof(ext_err));
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
            return trap.data.error.code;
        }
    }
}

VmResult vm_execute(VmState *vm) {
    if (!(vm->module->header.flags & NVM_FLAG_HAS_MAIN)) {
        return vm_error(vm, VM_ERR_UNDEFINED_FUNCTION, "No entry point defined");
    }

    uint32_t entry = vm->module->header.entry_point;
    if (entry >= vm->module->function_count) {
        return vm_error(vm, VM_ERR_UNDEFINED_FUNCTION, "Entry point %u out of range", entry);
    }

    return vm_call_function(vm, entry, NULL, 0);
}
