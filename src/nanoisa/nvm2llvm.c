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
static void float_runtime(FILE *out) {
    fputs(
        "define internal double @floating(%V %v) {\n"
        " %bits = call i64 @integer(%V %v, i8 3)\n %x = bitcast i64 %bits to double\n ret double %x\n}\n"
        "define internal i1 @truthy(%V %v) {\n"
        " %bits = extractvalue %V %v, 0\n %tag = extractvalue %V %v, 1\n"
        " %float = icmp eq i8 %tag, 3\n %x = bitcast i64 %bits to double\n"
        " %f = fcmp une double %x, 0.000000e+00\n %i = icmp ne i64 %bits, 0\n"
        " %answer = select i1 %float, i1 %f, i1 %i\n ret i1 %answer\n}\n"
        "define internal i64 @cast_integer(%V %v) {\nentry:\n"
        " %bits = extractvalue %V %v, 0\n %tag = extractvalue %V %v, 1\n"
        " %float = icmp eq i8 %tag, 3\n br i1 %float, label %fp, label %scalar\n"
        "fp:\n %x = bitcast i64 %bits to double\n"
        " %lower = fcmp oge double %x, 0xC3E0000000000000\n"
        " %upper = fcmp olt double %x, 0x43E0000000000000\n %valid = and i1 %lower, %upper\n"
        " call void @check(i1 %valid)\n %answer = fptosi double %x to i64\n ret i64 %answer\n"
        "scalar:\n ret i64 %bits\n}\n"
        "define internal double @cast_floating(%V %v) {\nentry:\n"
        " %bits = extractvalue %V %v, 0\n %tag = extractvalue %V %v, 1\n"
        " %float = icmp eq i8 %tag, 3\n br i1 %float, label %fp, label %scalar\n"
        "fp:\n %x = bitcast i64 %bits to double\n ret double %x\n"
        "scalar:\n %answer = sitofp i64 %bits to double\n ret double %answer\n}\n"
        "define internal double @float_divide(double %a, double %b) {\nentry:\n"
        " %zero = fcmp oeq double %b, 0.000000e+00\n br i1 %zero, label %z, label %divide\n"
        "z:\n ret double 0.000000e+00\ndivide:\n %answer = fdiv double %a, %b\n ret double %answer\n}\n", out);
}
/* I inspect both tags before promotion; cast_floating alone also accepts
 * nonnumeric scalar tags and is therefore not an arithmetic eligibility test. */
static void numeric_runtime(FILE *out) {
    const char *names[] = {"add", "sub", "mul", "div"};
    const char *integer_ops[] = {"add", "sub", "mul"};
    const char *float_ops[] = {"fadd", "fsub", "fmul"};
    for (unsigned op = 0; op < 4; op++) {
        fprintf(out, "define internal %%V @numeric_%s(%%V %%a, %%V %%b) {\nentry:\n", names[op]);
        fputs(" %at = extractvalue %V %a, 1\n %bt = extractvalue %V %b, 1\n"
              " %ai = icmp eq i8 %at, 1\n %bi = icmp eq i8 %bt, 1\n"
              " %af = icmp eq i8 %at, 3\n %bf = icmp eq i8 %bt, 3\n"
              " %an = or i1 %ai, %af\n %bn = or i1 %bi, %bf\n"
              " %valid = and i1 %an, %bn\n call void @check(i1 %valid)\n"
              " %integers = and i1 %ai, %bi\n"
              " br i1 %integers, label %integer, label %floating\ninteger:\n"
              " %ix = extractvalue %V %a, 0\n %iy = extractvalue %V %b, 0\n", out);
        if (op == 3) fputs(" %ir = call i64 @divide(i64 %ix, i64 %iy, i1 false)\n", out);
        else fprintf(out, " %%ir = %s i64 %%ix, %%iy\n", integer_ops[op]);
        fputs(" %iv = insertvalue %V zeroinitializer, i64 %ir, 0\n"
              " %result_int = insertvalue %V %iv, i8 1, 1\n ret %V %result_int\nfloating:\n"
              " %fx = call double @cast_floating(%V %a)\n"
              " %fy = call double @cast_floating(%V %b)\n", out);
        if (op == 3) fputs(" %fr = call double @float_divide(double %fx, double %fy)\n", out);
        else fprintf(out, " %%fr = %s double %%fx, %%fy\n", float_ops[op]);
        fputs(" %bits = bitcast double %fr to i64\n"
              " %fv = insertvalue %V zeroinitializer, i64 %bits, 0\n"
              " %result_float = insertvalue %V %fv, i8 3, 1\n ret %V %result_float\n}\n", out);
    }
    fputs("define internal %V @numeric_neg(%V %a) {\nentry:\n"
          " %tag = extractvalue %V %a, 1\n %is_int = icmp eq i8 %tag, 1\n"
          " %is_float = icmp eq i8 %tag, 3\n %valid = or i1 %is_int, %is_float\n"
          " call void @check(i1 %valid)\n"
          " %bits = extractvalue %V %a, 0\n"
          " br i1 %is_int, label %integer, label %floating\ninteger:\n"
          " %ir = sub i64 0, %bits\n %iv = insertvalue %V %a, i64 %ir, 0\n ret %V %iv\n"
          "floating:\n %x = bitcast i64 %bits to double\n %r = fneg double %x\n"
          " %rb = bitcast double %r to i64\n %fv = insertvalue %V %a, i64 %rb, 0\n ret %V %fv\n}\n", out);
}
/* I preserve generic three-way NaN ordering independently of IEEE equality. */
static void comparison_runtime(FILE *out) {
    fputs(
        "define internal i1 @comparison_float_pair(%V %a, %V %b) {\n"
        " %at = extractvalue %V %a, 1\n"
        " %bt = extractvalue %V %b, 1\n"
        " %af = icmp eq i8 %at, 3\n"
        " %bf = icmp eq i8 %bt, 3\n"
        " %ai = icmp eq i8 %at, 1\n"
        " %bi = icmp eq i8 %bt, 1\n"
        " %an = or i1 %af, %ai\n"
        " %bn = or i1 %bf, %bi\n"
        " %left = and i1 %af, %bn\n"
        " %right = and i1 %bf, %an\n"
        " %answer = or i1 %left, %right\n"
        " ret i1 %answer\n"
        "}\n"
        "define internal i1 @scalar_equal(%V %a, %V %b) {\n"
        "entry:\n"
        " %fp = call i1 @comparison_float_pair(%V %a, %V %b)\n"
        " br i1 %fp, label %floating, label %tagged\n"
        "floating:\n"
        " %x = call double @cast_floating(%V %a)\n"
        " %y = call double @cast_floating(%V %b)\n"
        " %numeric = fcmp oeq double %x, %y\n"
        " ret i1 %numeric\n"
        "tagged:\n"
        " %at = extractvalue %V %a, 1\n"
        " %bt = extractvalue %V %b, 1\n"
        " %same = icmp eq i8 %at, %bt\n"
        " %av = extractvalue %V %a, 0\n"
        " %bv = extractvalue %V %b, 0\n"
        " %payload = icmp eq i64 %av, %bv\n"
        " %void = icmp eq i8 %at, 0\n"
        " %value = or i1 %void, %payload\n"
        " %answer = and i1 %same, %value\n"
        " ret i1 %answer\n"
        "}\n"
        "define internal i64 @scalar_order(%V %a, %V %b) {\n"
        "entry:\n"
        " %fp = call i1 @comparison_float_pair(%V %a, %V %b)\n"
        " br i1 %fp, label %floating, label %tagged\n"
        "floating:\n"
        " %x = call double @cast_floating(%V %a)\n"
        " %y = call double @cast_floating(%V %b)\n"
        " %lt = fcmp olt double %x, %y\n"
        " %gt = fcmp ogt double %x, %y\n"
        " %lo = zext i1 %lt to i64\n"
        " %hi = zext i1 %gt to i64\n"
        " %numeric = sub i64 %hi, %lo\n"
        " ret i64 %numeric\n"
        "tagged:\n"
        " %at = extractvalue %V %a, 1\n"
        " %bt = extractvalue %V %b, 1\n"
        " %same = icmp eq i8 %at, %bt\n"
        " %av = extractvalue %V %a, 0\n"
        " %bv = extractvalue %V %b, 0\n"
        " %less = icmp slt i64 %av, %bv\n"
        " %greater = icmp sgt i64 %av, %bv\n"
        " %l = zext i1 %less to i64\n"
        " %g = zext i1 %greater to i64\n"
        " %payload = sub i64 %g, %l\n"
        " %void = icmp eq i8 %at, 0\n"
        " %value = select i1 %void, i64 0, i64 %payload\n"
        " %ati = zext i8 %at to i64\n"
        " %bti = zext i8 %bt to i64\n"
        " %tags = sub i64 %ati, %bti\n"
        " %answer = select i1 %same, i64 %value, i64 %tags\n"
        " ret i64 %answer\n"
        "}\n"
        , out);
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
    fprintf(out, "define internal %s @f%u(", f->result_count ? "%V" : "void", index);
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
        case OP_PUSH_U8: case OP_PUSH_I64: case OP_PUSH_BOOL: case OP_PUSH_VOID: case OP_PUSH_F64: {
            int64_t float_bits = 0;
            if (ins.opcode == OP_PUSH_F64) memcpy(&float_bits, &ins.operands[0].f64, sizeof float_bits);
            fprintf(out, " call void @push(ptr %%stack, ptr %%sp, %%V { i64 %" PRId64 ", i8 %u })\n",
                ins.opcode == OP_PUSH_U8 ? (int64_t)ins.operands[0].u8 : ins.opcode == OP_PUSH_F64 ? float_bits : ins.opcode == OP_PUSH_I64 ? ins.operands[0].i64 : ins.opcode == OP_PUSH_BOOL ? (int64_t)(ins.operands[0].u8 != 0) : 0,
                ins.opcode == OP_PUSH_U8 ? TAG_U8 : ins.opcode == OP_PUSH_F64 ? TAG_FLOAT : ins.opcode == OP_PUSH_I64 ? TAG_INT : ins.opcode == OP_PUSH_BOOL ? TAG_BOOL : TAG_VOID);
            break;
        }
        case OP_POP: pop(out, pc, "a"); break;
        case OP_DUP: pop(out, pc, "a"); push(out, pc, "a"); push(out, pc, "a"); break;
        case OP_SWAP: pop(out, pc, "b"); pop(out, pc, "a"); push(out, pc, "b"); push(out, pc, "a"); break;
        case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
            fprintf(out, " %%p%u_local = getelementptr %%V, ptr %%locals, i64 %u\n", pc, ins.operands[0].u16);
            if (ins.opcode == OP_LOAD_LOCAL) {
                fprintf(out, " %%p%u_a = load %%V, ptr %%p%u_local\n", pc, pc); push(out, pc, "a");
            } else { pop(out, pc, "a"); fprintf(out, " store %%V %%p%u_a, ptr %%p%u_local\n", pc, pc); }
            break;
        case OP_LOAD_GLOBAL: case OP_STORE_GLOBAL:
            fprintf(out, " %%p%u_global = getelementptr %%V, ptr @globals, i64 %u\n",
                    pc, ins.operands[0].u32);
            if (ins.opcode == OP_LOAD_GLOBAL) {
                fprintf(out, " %%p%u_a = load %%V, ptr %%p%u_global\n", pc, pc);
                push(out, pc, "a");
            } else {
                pop(out, pc, "a");
                fprintf(out, " store %%V %%p%u_a, ptr %%p%u_global\n", pc, pc);
            }
            break;
        case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: {
            uint32_t target = (uint32_t)((int64_t)pc + ins.operands[0].i32);
            if (ins.opcode == OP_JMP) fprintf(out, " br label %%b%u\n", target);
            else {
                pop(out, pc, "a");
                fprintf(out, " %%p%u_cond = call i1 @truthy(%%V %%p%u_a)\n"
                    " br i1 %%p%u_cond, label %%b%u, label %%b%u\n", pc, pc, pc,
                    ins.opcode == OP_JMP_TRUE ? target : next, ins.opcode == OP_JMP_TRUE ? next : target);
            }
            terminates = 1; break;
        }
        case OP_CALL: {
            uint32_t callee = ins.operands[0].u32;
            for (uint16_t i = m->functions[callee].arity; i > 0; --i)
                fprintf(out, " %%p%u_arg%u = call %%V @pop(ptr %%stack, ptr %%sp)\n", pc, i - 1);
            if (m->functions[callee].result_count)
                fprintf(out, " %%p%u_a = call %%V @f%u(", pc, callee);
            else fprintf(out, " call void @f%u(", callee);
            for (uint16_t i = 0; i < m->functions[callee].arity; ++i)
                fprintf(out, "%s%%V %%p%u_arg%u", i ? ", " : "", pc, i);
            fputs(")\n", out);
            if (m->functions[callee].result_count) push(out, pc, "a");
            break;
        }
        case OP_RET:
            fputs(" br label %return_result\n", out);
            terminates = 1; break;
        case OP_ASSERT:
            pop(out, pc, "a");
            fprintf(out, " %%p%u_ok = call i1 @truthy(%%V %%p%u_a)\n call void @check(i1 %%p%u_ok)\n", pc, pc, pc);
            break;
        case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
            pop(out, pc, "b");
            pop(out, pc, "a");
            if (ins.opcode == OP_EQ || ins.opcode == OP_NE) {
                fprintf(out, " %%p%u_equal = call i1 @scalar_equal(%%V %%p%u_a, %%V %%p%u_b)\n"
                    " %%p%u_bool = xor i1 %%p%u_equal, %s\n", pc, pc, pc, pc, pc,
                    ins.opcode == OP_NE ? "true" : "false");
            } else {
                const char *predicate = ins.opcode == OP_LT ? "slt" : ins.opcode == OP_LE ? "sle" :
                                        ins.opcode == OP_GT ? "sgt" : "sge";
                fprintf(out, " %%p%u_order = call i64 @scalar_order(%%V %%p%u_a, %%V %%p%u_b)\n"
                    " %%p%u_bool = icmp %s i64 %%p%u_order, 0\n", pc, pc, pc, pc, predicate, pc);
            }
            fprintf(out, " %%p%u_result = zext i1 %%p%u_bool to i64\n", pc, pc);
            result(out, pc, TAG_BOOL);
            break;
        }
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_NEG: {
            const char *name = ins.opcode == OP_ADD ? "add" : ins.opcode == OP_SUB ? "sub" :
                               ins.opcode == OP_MUL ? "mul" : ins.opcode == OP_DIV ? "div" : "neg";
            if (ins.opcode != OP_NEG) pop(out, pc, "b");
            pop(out, pc, "a");
            fprintf(out, " %%p%u_value = call %%V @numeric_%s(%%V %%p%u_a", pc, name, pc);
            if (ins.opcode != OP_NEG) fprintf(out, ", %%V %%p%u_b", pc);
            fputs(")\n", out);
            push(out, pc, "value");
            break;
        }
        case OP_MOD:
            pop(out, pc, "b");
            pop(out, pc, "a");
            fprintf(out, " %%p%u_x = call i64 @integer(%%V %%p%u_a, i8 1)\n"
                         " %%p%u_y = call i64 @integer(%%V %%p%u_b, i8 1)\n"
                         " %%p%u_result = call i64 @divide(i64 %%p%u_x, i64 %%p%u_y, i1 true)\n",
                    pc, pc, pc, pc, pc, pc, pc);
            result(out, pc, TAG_INT);
            break;
        case OP_CAST_BOOL: case OP_AND: case OP_OR: case OP_NOT: {
            int binary = ins.opcode == OP_AND || ins.opcode == OP_OR;
            if (binary) {
                pop(out, pc, "b");
                fprintf(out, " %%p%u_right = call i1 @truthy(%%V %%p%u_b)\n", pc, pc);
            }
            pop(out, pc, "a");
            fprintf(out, " %%p%u_left = call i1 @truthy(%%V %%p%u_a)\n", pc, pc);
            if (binary)
                fprintf(out, " %%p%u_bool = %s i1 %%p%u_left, %%p%u_right\n",
                        pc, ins.opcode == OP_AND ? "and" : "or", pc, pc);
            else
                fprintf(out, " %%p%u_bool = xor i1 %%p%u_left, %s\n", pc, pc,
                        ins.opcode == OP_NOT ? "true" : "false");
            fprintf(out, " %%p%u_result = zext i1 %%p%u_bool to i64\n", pc, pc);
            result(out, pc, TAG_BOOL);
            break;
        }
        case OP_CAST_INT: case OP_CAST_FLOAT:
            pop(out, pc, "a");
            if (ins.opcode == OP_CAST_FLOAT) {
                fprintf(out, " %%p%u_fp = call double @cast_floating(%%V %%p%u_a)\n %%p%u_result = bitcast double %%p%u_fp to i64\n", pc, pc, pc, pc);
            } else {
                fprintf(out, " %%p%u_result = call i64 @cast_integer(%%V %%p%u_a)\n", pc, pc);
            }
            result(out, pc, ins.opcode == OP_CAST_FLOAT ? TAG_FLOAT : TAG_INT);
            break;
        case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV:
        case OP_F64_NEG: case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT:
        case OP_F64_LE: case OP_F64_GT: case OP_F64_GE: {
            if (ins.opcode != OP_F64_NEG) pop(out, pc, "b");
            pop(out, pc, "a");
            fprintf(out, " %%p%u_x = call double @floating(%%V %%p%u_a)\n", pc, pc);
            if (ins.opcode != OP_F64_NEG) fprintf(out, " %%p%u_y = call double @floating(%%V %%p%u_b)\n", pc, pc);
            const char *op = NULL, *cmp = NULL;
            switch (ins.opcode) {
            case OP_F64_ADD: op="fadd"; break; case OP_F64_SUB: op="fsub"; break; case OP_F64_MUL: op="fmul"; break;
            case OP_F64_EQ: cmp="oeq"; break; case OP_F64_NE: cmp="une"; break;
            case OP_F64_LT: cmp="olt"; break; case OP_F64_LE: cmp="ole"; break;
            case OP_F64_GT: cmp="ogt"; break; case OP_F64_GE: cmp="oge"; break;
            default: break;
            }
            if (cmp) {
                fprintf(out, " %%p%u_cmp = fcmp %s double %%p%u_x, %%p%u_y\n %%p%u_result = zext i1 %%p%u_cmp to i64\n", pc, cmp, pc, pc, pc, pc);
            } else {
                if (op) fprintf(out, " %%p%u_fp = %s double %%p%u_x, %%p%u_y\n", pc, op, pc, pc);
                else if (ins.opcode == OP_F64_NEG) fprintf(out, " %%p%u_fp = fneg double %%p%u_x\n", pc, pc);
                else fprintf(out, " %%p%u_fp = call double @float_divide(double %%p%u_x, double %%p%u_y)\n", pc, pc, pc);
                fprintf(out, " %%p%u_result = bitcast double %%p%u_fp to i64\n", pc, pc);
            }
            result(out, pc, cmp ? TAG_BOOL : TAG_FLOAT);
            break;
        }
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
    fprintf(out, "b%u:\n br label %%return_result\nreturn_result:\n"
        " %%result_count = load i64, ptr %%sp\n %%result_shape = icmp eq i64 %%result_count, %u\n"
        " call void @check(i1 %%result_shape)\n", f->code_length, f->result_count);
    if (f->result_count)
        fprintf(out, " %%returned = call %%V @pop(ptr %%stack, ptr %%sp)\n"
            " call i64 @integer(%%V %%returned, i8 %u)\n ret %%V %%returned\n}\n", f->result_tag);
    else fputs(" ret void\n}\n", out);
}
int nvm2llvm_emit_entry(const NvmModule *m, FILE *out, char *error, size_t size, const char *entry) {
    if (!entry || (strcmp(entry, "main") && strncmp(entry, "nano_", 5)))
        return refuse(error, size, "I require main or a nano_ entry identifier");
    for (const char *p = entry; *p; ++p)
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
              (*p >= '0' && *p <= '9') || *p == '_'))
            return refuse(error, size, "I require an ASCII entry identifier");
    if (!m || !out) return refuse(error, size, "I require a module and output stream");
    NvmVerifyResult verified = nvm_verify_profile(m, NVM_PROFILE_CLOSED_SCALAR);
    if (!verified.ok) return refuse(error, size, "%s", verified.error_msg);
    /* I size storage from every verified literal global operand, matching
     * VM module allocation. Verification bounds index+1 by NVM_MAX_GLOBALS. */
    uint32_t global_count = 0;
    uint32_t initializer = m->function_count;
    for (uint32_t i = 0; i < m->function_count; ++i) {
        const NvmFunctionEntry *f = &m->functions[i];
        const char *name = nvm_get_string(m, f->name_idx);
        if (initializer == m->function_count && name && !strcmp(name, "__init__"))
            initializer = i;
        for (uint32_t pc = 0; pc < f->code_length;) {
            DecodedInstruction ins = {0};
            uint32_t width = isa_decode(m->code + f->code_offset + pc, f->code_length - pc, &ins);
            if (ins.opcode == OP_LOAD_GLOBAL || ins.opcode == OP_STORE_GLOBAL) {
                uint32_t count = ins.operands[0].u32 + 1;
                if (count > global_count) global_count = count;
            }
            pc += width;
        }
    }
    runtime(out);
    if (global_count)
        fprintf(out, "@globals = internal global [%u x %%V] zeroinitializer\n", global_count);
    float_runtime(out);
    numeric_runtime(out);
    comparison_runtime(out);
    for (uint32_t i = 0; i < m->function_count; ++i) {
        uint16_t depth = 0;
        verified = nvm_verify_function_max_stack(m, i, &depth);
        if (!verified.ok) return refuse(error, size, "I cannot establish scalar stack depth");
        function(out, m, i, depth);
    }
    fprintf(out, "define i32 @%s() {\n", entry);
    if (initializer < m->function_count) {
        if (m->functions[initializer].result_count)
            fprintf(out, " %%initialized = call %%V @f%u()\n", initializer);
        else
            fprintf(out, " call void @f%u()\n", initializer);
    }
    fprintf(out, " %%value = call %%V @f%u()\n %%n = extractvalue %%V %%value, 0\n %%status = trunc i64 %%n to i32\n ret i32 %%status\n}\n", m->header.entry_point);
    if (ferror(out)) return refuse(error, size, "I could not write LLVM IR");
    return 1;
}

int nvm2llvm_emit(const NvmModule *m, FILE *out, char *error, size_t size) {
    return nvm2llvm_emit_entry(m, out, error, size, "main");
}
