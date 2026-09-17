/* I compile scalar instructions to LLVM blocks, never to a bytecode dispatcher. */
#include "nvm2llvm.h"
#include "verifier.h"
#include <inttypes.h>
#include <stdarg.h>
#include <string.h>

static int refuse(char *error, size_t size, const char *format, ...) {
    va_list ap;
    va_start(ap, format); vsnprintf(error, size, format, ap); va_end(ap);
    return 0;
}
static int scalar(uint8_t tag) { return tag == TAG_INT || tag == TAG_BOOL || tag == TAG_VOID; }
static int supported(uint8_t op) {
    switch (op) {
    case OP_NOP: case OP_PUSH_I64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
    case OP_DUP: case OP_POP: case OP_SWAP: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S: case OP_I64_REM_S:
    case OP_I64_NEG: case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S:
    case OP_I64_GT_S: case OP_I64_GE_S: case OP_BOOL_AND: case OP_BOOL_OR: case OP_BOOL_NOT:
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: case OP_CALL: case OP_RET:
    case OP_ASSERT: case OP_TYPE_CHECK: return 1;
    default: return 0;
    }
}
static void runtime(FILE *out) {
    fputs("; I compile verified scalar NanoISA directly.\n"
        "%V = type { i64, i8 }\n"
        "declare void @llvm.trap() cold noreturn nounwind\n"
        "define internal void @check(i1 %ok) {\nentry:\n br i1 %ok, label %done, label %bad\nbad:\n call void @llvm.trap()\n unreachable\ndone:\n ret void\n}\n"
        "define internal i64 @integer(%V %v, i8 %expected) {\n"
        " %tag = extractvalue %V %v, 1\n %ok = icmp eq i8 %tag, %expected\n"
        " call void @check(i1 %ok)\n %x = extractvalue %V %v, 0\n ret i64 %x\n}\n"
        "define internal %V @pop(ptr %stack, ptr %sp) {\n"
        " %old = load i64, ptr %sp\n %n = sub i64 %old, 1\n store i64 %n, ptr %sp\n"
        " %p = getelementptr %V, ptr %stack, i64 %n\n %v = load %V, ptr %p\n ret %V %v\n}\n"
        "define internal void @push(ptr %stack, ptr %sp, %V %v) {\n"
        " %n = load i64, ptr %sp\n %p = getelementptr %V, ptr %stack, i64 %n\n"
        " store %V %v, ptr %p\n %next = add i64 %n, 1\n store i64 %next, ptr %sp\n ret void\n}\n"
        "define internal i64 @divide(i64 %a, i64 %b, i1 %rem) {\nentry:\n"
        " %zero = icmp eq i64 %b, 0\n br i1 %zero, label %z, label %nonzero\n"
        "z:\n ret i64 0\nnonzero:\n %min = icmp eq i64 %a, -9223372036854775808\n"
        " %minus = icmp eq i64 %b, -1\n %overflow = and i1 %min, %minus\n"
        " br i1 %overflow, label %wrap, label %safe\nwrap:\n"
        " %wrapped = select i1 %rem, i64 0, i64 -9223372036854775808\n ret i64 %wrapped\n"
        "safe:\n %q = sdiv i64 %a, %b\n %r = srem i64 %a, %b\n"
        " %answer = select i1 %rem, i64 %r, i64 %q\n ret i64 %answer\n}\n", out);
}
static void pop(FILE *out, uint32_t pc, const char *name) {
    fprintf(out, " %%p%u_%s = call %%V @pop(ptr %%stack, ptr %%sp)\n", pc, name);
}
static void push(FILE *out, uint32_t pc, const char *name) {
    fprintf(out, " call void @push(ptr %%stack, ptr %%sp, %%V %%p%u_%s)\n", pc, name);
}
static void result(FILE *out, uint32_t pc, uint8_t tag) {
    fprintf(out, " %%p%u_v0 = insertvalue %%V zeroinitializer, i64 %%p%u_result, 0\n"
        " %%p%u_v = insertvalue %%V %%p%u_v0, i8 %u, 1\n", pc, pc, pc, pc, tag);
    push(out, pc, "v");
}
static void function(FILE *out, const NvmModule *m, uint32_t index, uint16_t depth) {
    const NvmFunctionEntry *f = &m->functions[index];
    fprintf(out, "define internal %%V @f%u(", index);
    for (uint16_t i = 0; i < f->arity; ++i) fprintf(out, "%s%%V %%arg%u", i ? ", " : "", i);
    fprintf(out, ") {\nentry:\n %%stack = alloca [%u x %%V]\n %%sp = alloca i64\n"
        " store i64 0, ptr %%sp\n %%locals = alloca [%u x %%V]\n"
        " store [%u x %%V] zeroinitializer, ptr %%locals\n", depth ? depth : 1,
        f->local_count ? f->local_count : 1, f->local_count ? f->local_count : 1);
    for (uint16_t i = 0; i < f->arity; ++i)
        fprintf(out, " %%argp%u = getelementptr %%V, ptr %%locals, i64 %u\n store %%V %%arg%u, ptr %%argp%u\n", i, i, i, i);
    fputs(" br label %b0\n", out);
    for (uint32_t pc = 0; pc < f->code_length;) {
        DecodedInstruction ins = {0};
        uint32_t width = isa_decode(m->code + f->code_offset + pc, f->code_length - pc, &ins);
        uint32_t next = pc + width;
        int terminates = 0;
        fprintf(out, "b%u:\n", pc);
        switch (ins.opcode) {
        case OP_NOP: break;
        case OP_PUSH_I64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
            fprintf(out, " call void @push(ptr %%stack, ptr %%sp, %%V { i64 %" PRId64 ", i8 %u })\n",
                ins.opcode == OP_PUSH_I64 ? ins.operands[0].i64 : ins.opcode == OP_PUSH_BOOL ? (int64_t)(ins.operands[0].u8 != 0) : 0,
                ins.opcode == OP_PUSH_I64 ? TAG_INT : ins.opcode == OP_PUSH_BOOL ? TAG_BOOL : TAG_VOID);
            break;
        case OP_POP: pop(out, pc, "a"); break;
        case OP_DUP: pop(out, pc, "a"); push(out, pc, "a"); push(out, pc, "a"); break;
        case OP_SWAP: pop(out, pc, "b"); pop(out, pc, "a"); push(out, pc, "b"); push(out, pc, "a"); break;
        case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
            fprintf(out, " %%p%u_local = getelementptr %%V, ptr %%locals, i64 %u\n", pc, ins.operands[0].u16);
            if (ins.opcode == OP_LOAD_LOCAL) {
                fprintf(out, " %%p%u_a = load %%V, ptr %%p%u_local\n", pc, pc); push(out, pc, "a");
            } else { pop(out, pc, "a"); fprintf(out, " store %%V %%p%u_a, ptr %%p%u_local\n", pc, pc); }
            break;
        case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: {
            uint32_t target = (uint32_t)((int64_t)pc + ins.operands[0].i32);
            if (ins.opcode == OP_JMP) fprintf(out, " br label %%b%u\n", target);
            else {
                pop(out, pc, "a");
                fprintf(out, " %%p%u_x = extractvalue %%V %%p%u_a, 0\n %%p%u_cond = icmp ne i64 %%p%u_x, 0\n"
                    " br i1 %%p%u_cond, label %%b%u, label %%b%u\n", pc, pc, pc, pc, pc,
                    ins.opcode == OP_JMP_TRUE ? target : next, ins.opcode == OP_JMP_TRUE ? next : target);
            }
            terminates = 1; break;
        }
        case OP_CALL: {
            uint32_t callee = ins.operands[0].u32;
            for (uint16_t i = m->functions[callee].arity; i > 0; --i)
                fprintf(out, " %%p%u_arg%u = call %%V @pop(ptr %%stack, ptr %%sp)\n", pc, i - 1);
            fprintf(out, " %%p%u_a = call %%V @f%u(", pc, callee);
            for (uint16_t i = 0; i < m->functions[callee].arity; ++i)
                fprintf(out, "%s%%V %%p%u_arg%u", i ? ", " : "", pc, i);
            fputs(")\n", out); push(out, pc, "a"); break;
        }
        case OP_RET: pop(out, pc, "a"); fprintf(out, " ret %%V %%p%u_a\n", pc); terminates = 1; break;
        case OP_ASSERT:
            pop(out, pc, "a");
            fprintf(out, " %%p%u_x = extractvalue %%V %%p%u_a, 0\n %%p%u_ok = icmp ne i64 %%p%u_x, 0\n call void @check(i1 %%p%u_ok)\n", pc, pc, pc, pc, pc);
            break;
        case OP_TYPE_CHECK:
            pop(out, pc, "a");
            fprintf(out, " %%p%u_tag = extractvalue %%V %%p%u_a, 1\n %%p%u_cmp = icmp eq i8 %%p%u_tag, %u\n"
                " %%p%u_result = zext i1 %%p%u_cmp to i64\n", pc, pc, pc, pc, ins.operands[0].u8, pc, pc);
            result(out, pc, TAG_BOOL); break;
        default: {
            int unary = ins.opcode == OP_I64_NEG || ins.opcode == OP_BOOL_NOT;
            int boolean = ins.opcode == OP_BOOL_NOT || ins.opcode == OP_BOOL_AND || ins.opcode == OP_BOOL_OR;
            if (!unary) pop(out, pc, "b");
            pop(out, pc, "a");
            fprintf(out, " %%p%u_x = call i64 @integer(%%V %%p%u_a, i8 %u)\n", pc, pc, boolean ? TAG_BOOL : TAG_INT);
            if (!unary) fprintf(out, " %%p%u_y = call i64 @integer(%%V %%p%u_b, i8 %u)\n", pc, pc, boolean ? TAG_BOOL : TAG_INT);
            const char *op = NULL, *comparison = NULL;
            switch (ins.opcode) {
            case OP_I64_ADD: op="add"; break; case OP_I64_SUB: op="sub"; break; case OP_I64_MUL: op="mul"; break;
            case OP_BOOL_AND: op="and"; break; case OP_BOOL_OR: op="or"; break;
            case OP_I64_EQ: comparison="eq"; break; case OP_I64_NE: comparison="ne"; break;
            case OP_I64_LT_S: comparison="slt"; break; case OP_I64_LE_S: comparison="sle"; break;
            case OP_I64_GT_S: comparison="sgt"; break; case OP_I64_GE_S: comparison="sge"; break;
            default: break;
            }
            if (op) fprintf(out, " %%p%u_result = %s i64 %%p%u_x, %%p%u_y\n", pc, op, pc, pc);
            else if (comparison) fprintf(out, " %%p%u_cmp = icmp %s i64 %%p%u_x, %%p%u_y\n %%p%u_result = zext i1 %%p%u_cmp to i64\n", pc, comparison, pc, pc, pc, pc);
            else if (ins.opcode == OP_I64_NEG) fprintf(out, " %%p%u_result = sub i64 0, %%p%u_x\n", pc, pc);
            else if (ins.opcode == OP_BOOL_NOT) fprintf(out, " %%p%u_result = xor i64 %%p%u_x, 1\n", pc, pc);
            else fprintf(out, " %%p%u_result = call i64 @divide(i64 %%p%u_x, i64 %%p%u_y, i1 %s)\n", pc, pc, pc, ins.opcode == OP_I64_REM_S ? "true" : "false");
            result(out, pc, boolean || comparison ? TAG_BOOL : TAG_INT);
            break;
        }
        }
        if (!terminates) fprintf(out, " br label %%b%u\n", next);
        pc = next;
    }
    fprintf(out, "b%u:\n unreachable\n}\n", f->code_length);
}
int nvm2llvm_emit(const NvmModule *m, FILE *out, char *error, size_t size) {
    if (!m || !out) return refuse(error, size, "I require a module and output stream");
    NvmVerifyResult verified = nvm_verify(m);
    if (!verified.ok) return refuse(error, size, "I refuse unverified bytecode: %s", verified.error_msg);
    if (m->import_count || m->module_ref_count || m->struct_count || m->enum_count || m->union_count ||
        m->ownership_size || m->passive_size || m->layout_size)
        return refuse(error, size, "I support only closed scalar modules without imports, nominal layouts or ownership/passive contracts");
    if (m->functions[m->header.entry_point].arity)
        return refuse(error, size, "I require a zero-argument scalar entry point");
    for (uint32_t i = 0; i < m->function_count; ++i) {
        const NvmFunctionEntry *f = &m->functions[i];
        if (f->upvalue_count || f->result_count != 1 || (f->result_tag != TAG_INT && f->result_tag != TAG_BOOL))
            return refuse(error, size, "I require one integer/bool result and no captures in function %u", i);
        for (uint16_t p = 0; p < f->arity; ++p)
            if (m->function_param_types && m->function_param_types[i] && !scalar(m->function_param_types[i][p]))
                return refuse(error, size, "I require scalar parameters in function %u", i);
        for (uint32_t pc = 0; pc < f->code_length;) {
            DecodedInstruction ins = {0};
            uint32_t width = isa_decode(m->code + f->code_offset + pc, f->code_length - pc, &ins);
            if (!width || !supported(ins.opcode)) return refuse(error, size, "I do not support opcode 0x%02x at function %u offset %u in my scalar LLVM profile", ins.opcode, i, pc);
            pc += width;
        }
    }
    runtime(out);
    for (uint32_t i = 0; i < m->function_count; ++i) {
        uint16_t depth = 0;
        verified = nvm_verify_function_max_stack(m, i, &depth);
        if (!verified.ok) return refuse(error, size, "I cannot establish scalar stack depth");
        function(out, m, i, depth);
    }
    fprintf(out, "define i32 @main() {\n %%value = call %%V @f%u()\n %%n = extractvalue %%V %%value, 0\n %%status = trunc i64 %%n to i32\n ret i32 %%status\n}\n", m->header.entry_point);
    if (ferror(out)) return refuse(error, size, "I could not write LLVM IR");
    return 1;
}
