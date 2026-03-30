/*
 * wasm_backend.c — nanolang WASM binary emit backend
 *
 * Emits WebAssembly binary format (.wasm) from the nanolang AST.
 * Supports a pure-function subset: Int (i64), Float (f64), Bool (i32),
 * arithmetic, comparisons, function definitions, if/else, return.
 *
 * WASM binary format spec: https://webassembly.github.io/spec/core/binary/
 */

#include "wasm_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ── Dynamic byte buffer ─────────────────────────────────────────────── */
typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} WasmBuf;

static void buf_init(WasmBuf *b) {
    b->data = NULL; b->len = 0; b->cap = 0;
}
static void buf_free(WasmBuf *b) {
    free(b->data); b->data = NULL; b->len = b->cap = 0;
}
static void buf_grow(WasmBuf *b, size_t need) {
    if (b->len + need <= b->cap) return;
    size_t nc = b->cap ? b->cap * 2 : 256;
    while (nc < b->len + need) nc *= 2;
    b->data = realloc(b->data, nc);
    b->cap = nc;
}
static void buf_byte(WasmBuf *b, uint8_t v) {
    buf_grow(b, 1);
    b->data[b->len++] = v;
}
static void buf_bytes(WasmBuf *b, const uint8_t *src, size_t n) {
    buf_grow(b, n);
    memcpy(b->data + b->len, src, n);
    b->len += n;
}
static void buf_append(WasmBuf *dst, const WasmBuf *src) {
    buf_bytes(dst, src->data, src->len);
}

/* ── LEB128 encoding ─────────────────────────────────────────────────── */
static void emit_u32_leb(WasmBuf *b, uint32_t val) {
    do {
        uint8_t byte = val & 0x7F;
        val >>= 7;
        if (val) byte |= 0x80;
        buf_byte(b, byte);
    } while (val);
}
static void emit_i32_leb(WasmBuf *b, int32_t val) {
    int more = 1;
    while (more) {
        uint8_t byte = val & 0x7F;
        val >>= 7;
        if ((val == 0 && !(byte & 0x40)) || (val == -1 && (byte & 0x40)))
            more = 0;
        else
            byte |= 0x80;
        buf_byte(b, byte);
    }
}
static void emit_i64_leb(WasmBuf *b, int64_t val) {
    int more = 1;
    while (more) {
        uint8_t byte = val & 0x7F;
        val >>= 7;
        if ((val == 0 && !(byte & 0x40)) || (val == -1 && (byte & 0x40)))
            more = 0;
        else
            byte |= 0x80;
        buf_byte(b, byte);
    }
}

/* ── WASM value types ────────────────────────────────────────────────── */
#define WASM_I32 0x7F
#define WASM_I64 0x7E
#define WASM_F32 0x7D
#define WASM_F64 0x7C

/* ── WASM opcodes ────────────────────────────────────────────────────── */
#define OP_UNREACHABLE 0x00
#define OP_NOP         0x01
#define OP_BLOCK       0x02
#define OP_IF          0x04
#define OP_ELSE        0x05
#define OP_END         0x0B
#define OP_RETURN      0x0F
#define OP_CALL        0x10
#define OP_DROP        0x1A
#define OP_LOCAL_GET   0x20
#define OP_LOCAL_SET   0x21
#define OP_LOCAL_TEE   0x22
#define OP_I32_CONST   0x41
#define OP_I64_CONST   0x42
#define OP_F64_CONST   0x44
#define OP_I32_EQZ     0x45
#define OP_I32_EQ      0x46
#define OP_I32_NE      0x47
#define OP_I64_EQZ     0x50
#define OP_I64_EQ      0x51
#define OP_I64_NE      0x52
#define OP_I64_LT_S    0x53
#define OP_I64_GT_S    0x55
#define OP_I64_LE_S    0x57
#define OP_I64_GE_S    0x59
#define OP_F64_EQ      0x61
#define OP_F64_NE      0x62
#define OP_F64_LT      0x63
#define OP_F64_GT      0x64
#define OP_F64_LE      0x65
#define OP_F64_GE      0x66
#define OP_I64_ADD     0x7C
#define OP_I64_SUB     0x7D
#define OP_I64_MUL     0x7E
#define OP_I64_DIV_S   0x7F
#define OP_I64_REM_S   0x81
#define OP_F64_ADD     0xA0
#define OP_F64_SUB     0xA1
#define OP_F64_MUL     0xA2
#define OP_F64_DIV     0xA3
#define OP_I64_EXTEND_I32_S 0xAC

/* ── WASM section IDs ────────────────────────────────────────────────── */
#define SEC_TYPE    1
#define SEC_FUNC    3
#define SEC_EXPORT  7
#define SEC_CODE   10

/* ── Function info collected from AST ────────────────────────────────── */
typedef struct {
    const char *name;
    Parameter  *params;
    int         param_count;
    Type        return_type;
    ASTNode    *body;
    bool        is_extern;
    int         type_idx;      /* index into the type section */
    /* locals beyond params (from let bindings) */
    const char **local_names;
    int          local_count;
    int          local_cap;
} WasmFunc;

/* ── Compilation context ─────────────────────────────────────────────── */
typedef struct {
    WasmFunc *funcs;
    int       func_count;
    int       func_cap;
    bool      verbose;
    const char *error;
} WasmCtx;

static void ctx_init(WasmCtx *c, bool verbose) {
    memset(c, 0, sizeof(*c));
    c->verbose = verbose;
}
static void ctx_free(WasmCtx *c) {
    for (int i = 0; i < c->func_count; i++) {
        free(c->funcs[i].local_names);
    }
    free(c->funcs);
}

/* Map nanolang Type → WASM value type byte */
static uint8_t wasm_valtype(Type t) {
    switch (t) {
        case TYPE_INT:    return WASM_I64;
        case TYPE_FLOAT:  return WASM_F64;
        case TYPE_BOOL:   return WASM_I32;
        default:          return WASM_I64; /* fallback */
    }
}

/* ── Collect functions from the AST ──────────────────────────────────── */
static void collect_functions(WasmCtx *ctx, ASTNode *root) {
    if (!root) return;
    if (root->type == AST_PROGRAM) {
        for (int i = 0; i < root->as.program.count; i++)
            collect_functions(ctx, root->as.program.items[i]);
        return;
    }
    if (root->type != AST_FUNCTION) return;
    if (root->as.function.is_extern) return;

    if (ctx->func_count >= ctx->func_cap) {
        ctx->func_cap = ctx->func_cap ? ctx->func_cap * 2 : 16;
        ctx->funcs = realloc(ctx->funcs, ctx->func_cap * sizeof(WasmFunc));
    }
    WasmFunc *f = &ctx->funcs[ctx->func_count++];
    memset(f, 0, sizeof(*f));
    f->name        = root->as.function.name;
    f->params      = root->as.function.params;
    f->param_count = root->as.function.param_count;
    f->return_type = root->as.function.return_type;
    f->body        = root->as.function.body;
    f->is_extern   = root->as.function.is_extern;
}

/* Find the index of a local (param or let-binding) by name */
static int find_local(WasmFunc *f, const char *name) {
    /* Check params first */
    for (int i = 0; i < f->param_count; i++) {
        if (strcmp(f->params[i].name, name) == 0) return i;
    }
    /* Check additional locals */
    for (int i = 0; i < f->local_count; i++) {
        if (strcmp(f->local_names[i], name) == 0) return f->param_count + i;
    }
    return -1;
}

/* Add a new local variable, return its index */
static int add_local(WasmFunc *f, const char *name) {
    int idx = find_local(f, name);
    if (idx >= 0) return idx; /* already exists */
    if (f->local_count >= f->local_cap) {
        f->local_cap = f->local_cap ? f->local_cap * 2 : 8;
        f->local_names = realloc(f->local_names, f->local_cap * sizeof(const char *));
    }
    f->local_names[f->local_count] = name;
    return f->param_count + f->local_count++;
}

/* Find a function index by name */
static int find_func(WasmCtx *ctx, const char *name) {
    for (int i = 0; i < ctx->func_count; i++) {
        if (strcmp(ctx->funcs[i].name, name) == 0) return i;
    }
    return -1;
}

/* ── Emit expression bytecode ────────────────────────────────────────── */
static int emit_expr(WasmCtx *ctx, WasmFunc *func, WasmBuf *code, ASTNode *node);

static int emit_expr(WasmCtx *ctx, WasmFunc *func, WasmBuf *code, ASTNode *node) {
    if (!node) {
        ctx->error = "null AST node in expression";
        return -1;
    }
    switch (node->type) {
    case AST_NUMBER:
        buf_byte(code, OP_I64_CONST);
        emit_i64_leb(code, node->as.number);
        return 0;
    case AST_FLOAT:
        buf_byte(code, OP_F64_CONST);
        buf_grow(code, 8);
        memcpy(code->data + code->len, &node->as.float_val, 8);
        code->len += 8;
        return 0;
    case AST_BOOL:
        buf_byte(code, OP_I32_CONST);
        emit_i32_leb(code, node->as.bool_val ? 1 : 0);
        return 0;
    case AST_IDENTIFIER: {
        int idx = find_local(func, node->as.identifier);
        if (idx < 0) {
            ctx->error = "undefined variable in WASM backend";
            if (ctx->verbose) fprintf(stderr, "[wasm] undefined: %s\n", node->as.identifier);
            return -1;
        }
        buf_byte(code, OP_LOCAL_GET);
        emit_u32_leb(code, (uint32_t)idx);
        return 0;
    }
    case AST_PREFIX_OP: {
        /* Binary ops: (op lhs rhs) */
        if (node->as.prefix_op.arg_count != 2) {
            /* Unary minus: (- x) → 0 - x */
            if (node->as.prefix_op.arg_count == 1 &&
                node->as.prefix_op.op == TOKEN_MINUS) {
                buf_byte(code, OP_I64_CONST);
                emit_i64_leb(code, 0);
                if (emit_expr(ctx, func, code, node->as.prefix_op.args[0])) return -1;
                buf_byte(code, OP_I64_SUB);
                return 0;
            }
            ctx->error = "unsupported arity in prefix op";
            return -1;
        }
        ASTNode *lhs = node->as.prefix_op.args[0];
        ASTNode *rhs = node->as.prefix_op.args[1];
        if (emit_expr(ctx, func, code, lhs)) return -1;
        if (emit_expr(ctx, func, code, rhs)) return -1;
        switch (node->as.prefix_op.op) {
            case TOKEN_PLUS:     buf_byte(code, OP_I64_ADD); break;
            case TOKEN_MINUS:    buf_byte(code, OP_I64_SUB); break;
            case TOKEN_STAR:     buf_byte(code, OP_I64_MUL); break;
            case TOKEN_SLASH:    buf_byte(code, OP_I64_DIV_S); break;
            case TOKEN_PERCENT:  buf_byte(code, OP_I64_REM_S); break;
            case TOKEN_EQ:       buf_byte(code, OP_I64_EQ); break;
            case TOKEN_NE:       buf_byte(code, OP_I64_NE); break;
            case TOKEN_LT:       buf_byte(code, OP_I64_LT_S); break;
            case TOKEN_GT:       buf_byte(code, OP_I64_GT_S); break;
            case TOKEN_LE:       buf_byte(code, OP_I64_LE_S); break;
            case TOKEN_GE:       buf_byte(code, OP_I64_GE_S); break;
            default:
                ctx->error = "unsupported operator in WASM backend";
                return -1;
        }
        return 0;
    }
    case AST_CALL: {
        const char *name = node->as.call.name;
        int fidx = find_func(ctx, name);
        if (fidx < 0) {
            if (ctx->verbose) fprintf(stderr, "[wasm] unknown function: %s\n", name);
            ctx->error = "undefined function in WASM backend";
            return -1;
        }
        for (int i = 0; i < node->as.call.arg_count; i++) {
            if (emit_expr(ctx, func, code, node->as.call.args[i])) return -1;
        }
        buf_byte(code, OP_CALL);
        emit_u32_leb(code, (uint32_t)fidx);
        return 0;
    }
    case AST_IF: {
        /* Emit condition — result must be i32 for WASM if instruction.
         * Comparisons already yield i32. Booleans are i32. 
         * If condition is an i64 (e.g. raw integer), convert: (i64 != 0) */
        if (emit_expr(ctx, func, code, node->as.if_stmt.condition)) return -1;
        /* Condition type heuristic: if the node is a plain comparison/bool it
         * already yielded i32; if it's a number/identifier (i64) we need to
         * wrap it.  Emit i64.eqz + i32.eqz only for non-comparison nodes. */
        ASTNode *cond = node->as.if_stmt.condition;
        bool cond_is_i32 = (cond->type == AST_BOOL) ||
            (cond->type == AST_PREFIX_OP && (
                cond->as.prefix_op.op == TOKEN_EQ ||
                cond->as.prefix_op.op == TOKEN_NE ||
                cond->as.prefix_op.op == TOKEN_LT ||
                cond->as.prefix_op.op == TOKEN_GT ||
                cond->as.prefix_op.op == TOKEN_LE ||
                cond->as.prefix_op.op == TOKEN_GE));
        if (!cond_is_i32) {
            /* i64 → i32: truthy if != 0 */
            buf_byte(code, OP_I64_EQZ);
            buf_byte(code, OP_I32_EQZ);
        }
        buf_byte(code, OP_IF);
        buf_byte(code, WASM_I64); /* block type: returns i64 */
        if (emit_expr(ctx, func, code, node->as.if_stmt.then_branch)) return -1;
        if (node->as.if_stmt.else_branch) {
            buf_byte(code, OP_ELSE);
            if (emit_expr(ctx, func, code, node->as.if_stmt.else_branch)) return -1;
        } else {
            /* No else → push 0 as default */
            buf_byte(code, OP_ELSE);
            buf_byte(code, OP_I64_CONST);
            emit_i64_leb(code, 0);
        }
        buf_byte(code, OP_END);
        return 0;
    }
    case AST_BLOCK: {
        /* Emit all statements; the value of the last one is the block value */
        for (int i = 0; i < node->as.block.count; i++) {
            if (emit_expr(ctx, func, code, node->as.block.statements[i])) return -1;
            /* Drop intermediate values (not the last one) */
            if (i < node->as.block.count - 1) {
                ASTNode *s = node->as.block.statements[i];
                /* Statements that don't push a value: let, set */
                if (s->type != AST_LET && s->type != AST_SET)
                    buf_byte(code, OP_DROP);
            }
        }
        return 0;
    }
    case AST_LET: {
        /* Compile the value, store in a local */
        int idx = add_local(func, node->as.let.name);
        if (node->as.let.value) {
            if (emit_expr(ctx, func, code, node->as.let.value)) return -1;
        } else {
            buf_byte(code, OP_I64_CONST);
            emit_i64_leb(code, 0);
        }
        buf_byte(code, OP_LOCAL_SET);
        emit_u32_leb(code, (uint32_t)idx);
        return 0;
    }
    case AST_SET: {
        int idx = find_local(func, node->as.set.name);
        if (idx < 0) {
            ctx->error = "set: undefined variable";
            return -1;
        }
        if (emit_expr(ctx, func, code, node->as.set.value)) return -1;
        buf_byte(code, OP_LOCAL_SET);
        emit_u32_leb(code, (uint32_t)idx);
        return 0;
    }
    case AST_RETURN:
        if (node->as.return_stmt.value) {
            if (emit_expr(ctx, func, code, node->as.return_stmt.value)) return -1;
        }
        buf_byte(code, OP_RETURN);
        return 0;
    default:
        if (ctx->verbose) fprintf(stderr, "[wasm] unsupported AST node type %d\n", node->type);
        ctx->error = "unsupported AST node type for WASM backend";
        return -1;
    }
}

/* ── Write a WASM section ────────────────────────────────────────────── */
static void emit_section(WasmBuf *out, uint8_t sec_id, const WasmBuf *content) {
    buf_byte(out, sec_id);
    emit_u32_leb(out, (uint32_t)content->len);
    buf_append(out, content);
}

/* ── Main emit routine ───────────────────────────────────────────────── */
int wasm_backend_emit_fp(ASTNode *root, FILE *out, bool verbose) {
    WasmCtx ctx;
    ctx_init(&ctx, verbose);

    /* Pass 1: collect all top-level functions */
    collect_functions(&ctx, root);
    if (ctx.func_count == 0) {
        fprintf(stderr, "[wasm] error: no functions found to compile\n");
        ctx_free(&ctx);
        return 1;
    }
    if (verbose) fprintf(stderr, "[wasm] collected %d function(s)\n", ctx.func_count);

    /* Assign type indices (one type per function signature) */
    for (int i = 0; i < ctx.func_count; i++)
        ctx.funcs[i].type_idx = i;

    /* Pass 2: pre-scan function bodies for let bindings (to know local count) */
    /* (locals are added dynamically during emit, so we do a two-pass on code) */

    WasmBuf final_out;
    buf_init(&final_out);

    /* ── Magic + version ──────────────────────────────────────────────── */
    buf_bytes(&final_out, (const uint8_t[]){0x00, 0x61, 0x73, 0x6D}, 4); /* \0asm */
    buf_bytes(&final_out, (const uint8_t[]){0x01, 0x00, 0x00, 0x00}, 4); /* version 1 */

    /* ── Type section ─────────────────────────────────────────────────── */
    {
        WasmBuf sec;
        buf_init(&sec);
        emit_u32_leb(&sec, (uint32_t)ctx.func_count); /* num types */
        for (int i = 0; i < ctx.func_count; i++) {
            WasmFunc *f = &ctx.funcs[i];
            buf_byte(&sec, 0x60); /* func type */
            emit_u32_leb(&sec, (uint32_t)f->param_count);
            for (int p = 0; p < f->param_count; p++)
                buf_byte(&sec, wasm_valtype(f->params[p].type));
            /* Return type */
            if (f->return_type == TYPE_VOID) {
                emit_u32_leb(&sec, 0);
            } else {
                emit_u32_leb(&sec, 1);
                buf_byte(&sec, wasm_valtype(f->return_type));
            }
        }
        emit_section(&final_out, SEC_TYPE, &sec);
        buf_free(&sec);
    }

    /* ── Function section (maps func index → type index) ──────────────── */
    {
        WasmBuf sec;
        buf_init(&sec);
        emit_u32_leb(&sec, (uint32_t)ctx.func_count);
        for (int i = 0; i < ctx.func_count; i++)
            emit_u32_leb(&sec, (uint32_t)ctx.funcs[i].type_idx);
        emit_section(&final_out, SEC_FUNC, &sec);
        buf_free(&sec);
    }

    /* ── Export section (export all non-extern functions) ──────────────── */
    {
        WasmBuf sec;
        buf_init(&sec);
        int export_count = 0;
        for (int i = 0; i < ctx.func_count; i++)
            if (!ctx.funcs[i].is_extern) export_count++;
        emit_u32_leb(&sec, (uint32_t)export_count);
        for (int i = 0; i < ctx.func_count; i++) {
            if (ctx.funcs[i].is_extern) continue;
            size_t nlen = strlen(ctx.funcs[i].name);
            emit_u32_leb(&sec, (uint32_t)nlen);
            buf_bytes(&sec, (const uint8_t *)ctx.funcs[i].name, nlen);
            buf_byte(&sec, 0x00); /* export kind: func */
            emit_u32_leb(&sec, (uint32_t)i); /* func index */
        }
        emit_section(&final_out, SEC_EXPORT, &sec);
        buf_free(&sec);
    }

    /* ── Code section ─────────────────────────────────────────────────── */
    {
        WasmBuf sec;
        buf_init(&sec);
        emit_u32_leb(&sec, (uint32_t)ctx.func_count);
        for (int i = 0; i < ctx.func_count; i++) {
            WasmFunc *f = &ctx.funcs[i];
            WasmBuf body;
            buf_init(&body);

            /* Emit function body bytecode */
            WasmBuf code;
            buf_init(&code);
            if (emit_expr(&ctx, f, &code, f->body)) {
                fprintf(stderr, "[wasm] error compiling %s: %s\n", f->name, ctx.error ? ctx.error : "unknown");
                buf_free(&code);
                buf_free(&body);
                buf_free(&sec);
                buf_free(&final_out);
                ctx_free(&ctx);
                return 1;
            }
            buf_byte(&code, OP_END); /* function end */

            /* Now we know locals: emit local declarations then code */
            /* Locals: one group per type (all i64 for simplicity in v1) */
            if (f->local_count > 0) {
                emit_u32_leb(&body, 1); /* 1 group of locals */
                emit_u32_leb(&body, (uint32_t)f->local_count);
                buf_byte(&body, WASM_I64); /* all locals are i64 for now */
            } else {
                emit_u32_leb(&body, 0); /* no additional locals */
            }
            buf_append(&body, &code);
            buf_free(&code);

            /* Emit function body size + body */
            emit_u32_leb(&sec, (uint32_t)body.len);
            buf_append(&sec, &body);
            buf_free(&body);
        }
        emit_section(&final_out, SEC_CODE, &sec);
        buf_free(&sec);
    }

    /* Write to file */
    size_t written = fwrite(final_out.data, 1, final_out.len, out);
    if (written != final_out.len) {
        fprintf(stderr, "[wasm] error: short write (%zu/%zu bytes)\n", written, final_out.len);
        buf_free(&final_out);
        ctx_free(&ctx);
        return 1;
    }

    if (verbose) fprintf(stderr, "[wasm] emitted %zu bytes\n", final_out.len);

    buf_free(&final_out);
    ctx_free(&ctx);
    return 0;
}

int wasm_backend_emit(ASTNode *root, const char *output_path, bool verbose) {
    FILE *f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "[wasm] error: cannot open %s for writing\n", output_path);
        return 1;
    }
    int rc = wasm_backend_emit_fp(root, f, verbose);
    fclose(f);
    return rc;
}
