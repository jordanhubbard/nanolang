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

/* Return the number of bytes needed to encode val as unsigned LEB128 */
static size_t leb_u32_size(uint32_t val) {
    size_t n = 0;
    do { n++; val >>= 7; } while (val);
    return n;
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
#define SEC_CUSTOM  0

/* ── Source map tracking ─────────────────────────────────────────────── */

/* One mapping entry: function-relative code offset + source position */
typedef struct {
    int      func_idx;    /* index into ctx->funcs[] */
    uint32_t rel_offset;  /* byte offset within the function's code buffer */
    int      line;        /* 1-based source line */
    int      col;         /* 1-based source column */
} SrcMapEntry;

typedef struct {
    SrcMapEntry *entries;
    int          count, cap;
} SrcMap;

static void sm_init(SrcMap *s) { memset(s, 0, sizeof(*s)); }
static void sm_free(SrcMap *s) { free(s->entries); memset(s, 0, sizeof(*s)); }

static void sm_add(SrcMap *s, int fi, uint32_t off, int ln, int col) {
    if (ln <= 0) return;
    if (s->count >= s->cap) {
        s->cap = s->cap ? s->cap * 2 : 64;
        s->entries = realloc(s->entries, s->cap * sizeof(*s->entries));
    }
    s->entries[s->count++] = (SrcMapEntry){ fi, off, ln, col };
}

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
    /* source position of the function definition */
    int          src_line;
    int          src_col;
} WasmFunc;

/* ── Compilation context ─────────────────────────────────────────────── */
typedef struct {
    WasmFunc *funcs;
    int       func_count;
    int       func_cap;
    bool      verbose;
    const char *error;
    /* source map state (populated during emit_expr) */
    SrcMap    srcmap;
    int       cur_func_idx;  /* set before calling emit_expr for a function */
} WasmCtx;

static void ctx_init(WasmCtx *c, bool verbose) {
    memset(c, 0, sizeof(*c));
    c->verbose = verbose;
    sm_init(&c->srcmap);
}
static void ctx_free(WasmCtx *c) {
    for (int i = 0; i < c->func_count; i++) {
        free(c->funcs[i].local_names);
    }
    free(c->funcs);
    sm_free(&c->srcmap);
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
    f->src_line    = root->line;
    f->src_col     = root->column;
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
    /* Record source position before emitting this node */
    sm_add(&ctx->srcmap, ctx->cur_func_idx, (uint32_t)code->len,
           node->line, node->column);

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
    case AST_PAR_LET: {
        /* par-let: asyncify-compatible sequential evaluation.
         * Bindings are independent and could be scheduled in parallel by an agentOS runtime.
         * Here we emit sequentially and annotate with a custom section marker. */
        if (ctx->verbose)
            fprintf(stderr, "[wasm] par-let parallel_begin at line %d\n", node->line);
        for (int i = 0; i < node->as.par_let.count; i++) {
            int idx = add_local(func, node->as.par_let.names[i]);
            if (emit_expr(ctx, func, code, node->as.par_let.values[i])) return -1;
            buf_byte(code, OP_LOCAL_SET);
            emit_u32_leb(code, (uint32_t)idx);
        }
        if (ctx->verbose)
            fprintf(stderr, "[wasm] par-let parallel_end at line %d\n", node->line);
        /* Emit body expression (par-let result) */
        if (emit_expr(ctx, func, code, node->as.par_let.body)) return -1;
        return 0;
    }
    default:
        if (ctx->verbose) fprintf(stderr, "[wasm] unsupported AST node type %d\n", node->type);
        ctx->error = "unsupported AST node type for WASM backend";
        return -1;
    }
}

/* ── Write a WASM section ────────────────────────────────────────────── */

/* Forward declarations for the emit entry points */
int wasm_backend_emit_fp_ex(ASTNode *root, FILE *out, bool verbose,
                             const char *wasm_path,
                             const char *source_file,
                             const char *sourcemap_path);

static void emit_section(WasmBuf *out, uint8_t sec_id, const WasmBuf *content) {
    buf_byte(out, sec_id);
    emit_u32_leb(out, (uint32_t)content->len);
    buf_append(out, content);
}

/* Return the path's basename (pointer into path, no allocation) */
static const char *path_basename(const char *path) {
    const char *p = strrchr(path, '/');
    return p ? p + 1 : path;
}

/* Escape a string for JSON output (writes to fp) */
static void json_fwrite_str(FILE *fp, const char *s) {
    fputc('"', fp);
    for (; *s; s++) {
        switch (*s) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\n': fputs("\\n",  fp); break;
            case '\r': fputs("\\r",  fp); break;
            case '\t': fputs("\\t",  fp); break;
            default:   fputc(*s, fp);     break;
        }
    }
    fputc('"', fp);
}

/* ── Write .wasm.map JSON ────────────────────────────────────────────── */
static int write_source_map(const char *sourcemap_path,
                             const char *source_file,
                             WasmCtx *ctx,
                             const uint32_t *func_code_abs_offsets)
{
    FILE *fp = fopen(sourcemap_path, "w");
    if (!fp) {
        fprintf(stderr, "[wasm] error: cannot open %s for writing\n", sourcemap_path);
        return 1;
    }

    const char *source_base = path_basename(source_file);

    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": 1,\n");
    fprintf(fp, "  \"source\": "); json_fwrite_str(fp, source_base); fputs(",\n", fp);

    /* functions array: one entry per function with its definition site */
    fprintf(fp, "  \"functions\": [\n");
    for (int i = 0; i < ctx->func_count; i++) {
        WasmFunc *f = &ctx->funcs[i];
        fprintf(fp, "    {\"name\":");
        json_fwrite_str(fp, f->name);
        fprintf(fp, ",\"wasm_offset\":%u,\"src_line\":%d,\"src_col\":%d}",
                func_code_abs_offsets[i], f->src_line, f->src_col);
        if (i < ctx->func_count - 1) fputc(',', fp);
        fputc('\n', fp);
    }
    fprintf(fp, "  ],\n");

    /* instructions array: one entry per AST node emit point */
    fprintf(fp, "  \"instructions\": [\n");
    SrcMap *sm = &ctx->srcmap;
    bool first = true;
    for (int i = 0; i < sm->count; i++) {
        SrcMapEntry *e = &sm->entries[i];
        if (e->func_idx < 0 || e->func_idx >= ctx->func_count) continue;
        uint32_t abs_off = func_code_abs_offsets[e->func_idx] + e->rel_offset;
        if (!first) fputs(",\n", fp);
        first = false;
        fprintf(fp, "    {\"wasm_offset\":%u,\"src_line\":%d,\"src_col\":%d}",
                abs_off, e->line, e->col);
    }
    if (!first) fputc('\n', fp);
    fprintf(fp, "  ]\n}\n");

    fclose(fp);
    return 0;
}

/* ── Main emit routine ───────────────────────────────────────────────── */
int wasm_backend_emit_fp(ASTNode *root, FILE *out, bool verbose) {
    return wasm_backend_emit_fp_ex(root, out, verbose, NULL, NULL, NULL);
}

int wasm_backend_emit_fp_ex(ASTNode *root, FILE *out, bool verbose,
                             const char *wasm_path,
                             const char *source_file,
                             const char *sourcemap_path)
{
    WasmCtx ctx;
    ctx_init(&ctx, verbose);

    bool emit_srcmap = (sourcemap_path != NULL && source_file != NULL && wasm_path != NULL);

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
    /* Two-pass: build all function code/body buffers first so we know sizes,
     * then compute absolute code offsets for source map, then emit. */
    uint32_t *func_code_abs_offsets = NULL;
    {
        WasmBuf *all_codes  = calloc((size_t)ctx.func_count, sizeof(WasmBuf));
        WasmBuf *all_bodies = calloc((size_t)ctx.func_count, sizeof(WasmBuf));
        if (!all_codes || !all_bodies) {
            fprintf(stderr, "[wasm] out of memory\n");
            free(all_codes); free(all_bodies);
            buf_free(&final_out);
            ctx_free(&ctx);
            return 1;
        }

        /* Build code and body buffers for each function */
        for (int i = 0; i < ctx.func_count; i++) {
            WasmFunc *f = &ctx.funcs[i];
            buf_init(&all_codes[i]);
            buf_init(&all_bodies[i]);

            ctx.cur_func_idx = i;
            if (emit_expr(&ctx, f, &all_codes[i], f->body)) {
                fprintf(stderr, "[wasm] error compiling %s: %s\n",
                        f->name, ctx.error ? ctx.error : "unknown");
                for (int j = 0; j <= i; j++) {
                    buf_free(&all_codes[j]);
                    buf_free(&all_bodies[j]);
                }
                free(all_codes); free(all_bodies);
                buf_free(&final_out);
                ctx_free(&ctx);
                return 1;
            }
            buf_byte(&all_codes[i], OP_END); /* function end marker */

            /* Build body = locals_header + code */
            if (f->local_count > 0) {
                emit_u32_leb(&all_bodies[i], 1); /* 1 local group */
                emit_u32_leb(&all_bodies[i], (uint32_t)f->local_count);
                buf_byte(&all_bodies[i], WASM_I64); /* all locals are i64 for now */
            } else {
                emit_u32_leb(&all_bodies[i], 0); /* no additional locals */
            }
            buf_append(&all_bodies[i], &all_codes[i]);
        }

        /* Compute content size = leb(func_count) + Σ(leb(body[i].len) + body[i].len) */
        size_t content_size = leb_u32_size((uint32_t)ctx.func_count);
        for (int i = 0; i < ctx.func_count; i++)
            content_size += leb_u32_size((uint32_t)all_bodies[i].len) + all_bodies[i].len;

        /* Compute absolute code start offsets for source map:
         *   base = final_out.len + 1 (sec_id) + leb_size(content_size)
         *   For each function i, code starts at base + offset_within_content + body_leb_size + locals_size
         */
        if (emit_srcmap) {
            func_code_abs_offsets = calloc((size_t)ctx.func_count, sizeof(uint32_t));
            size_t base = final_out.len + 1 + leb_u32_size((uint32_t)content_size);
            size_t off  = leb_u32_size((uint32_t)ctx.func_count); /* past leb(func_count) */
            for (int i = 0; i < ctx.func_count; i++) {
                size_t body_leb  = leb_u32_size((uint32_t)all_bodies[i].len);
                size_t locals_sz = all_bodies[i].len - all_codes[i].len;
                func_code_abs_offsets[i] = (uint32_t)(base + off + body_leb + locals_sz);
                off += body_leb + all_bodies[i].len;
            }
        }

        /* Emit the code section */
        buf_byte(&final_out, SEC_CODE);
        emit_u32_leb(&final_out, (uint32_t)content_size);
        emit_u32_leb(&final_out, (uint32_t)ctx.func_count);
        for (int i = 0; i < ctx.func_count; i++) {
            emit_u32_leb(&final_out, (uint32_t)all_bodies[i].len);
            buf_append(&final_out, &all_bodies[i]);
            buf_free(&all_bodies[i]);
            buf_free(&all_codes[i]);
        }
        free(all_codes);
        free(all_bodies);
    }

    /* ── sourceMappingURL custom section ─────────────────────────────── */
    if (emit_srcmap) {
        /* WASM custom section 0x00: leb(name_len) + name + url_bytes */
        static const char sec_name[] = "sourceMappingURL";
        size_t name_len = sizeof(sec_name) - 1; /* 16 */
        const char *url  = path_basename(sourcemap_path);
        size_t url_len   = strlen(url);
        size_t content_sz = leb_u32_size((uint32_t)name_len) + name_len + url_len;

        buf_byte(&final_out, SEC_CUSTOM);
        emit_u32_leb(&final_out, (uint32_t)content_sz);
        emit_u32_leb(&final_out, (uint32_t)name_len);
        buf_bytes(&final_out, (const uint8_t *)sec_name, name_len);
        buf_bytes(&final_out, (const uint8_t *)url, url_len);
    }

    /* Write .wasm binary to file */
    size_t written = fwrite(final_out.data, 1, final_out.len, out);
    if (written != final_out.len) {
        fprintf(stderr, "[wasm] error: short write (%zu/%zu bytes)\n", written, final_out.len);
        free(func_code_abs_offsets);
        buf_free(&final_out);
        ctx_free(&ctx);
        return 1;
    }

    if (verbose) fprintf(stderr, "[wasm] emitted %zu bytes\n", final_out.len);

    buf_free(&final_out);

    /* Write .wasm.map JSON */
    int rc = 0;
    if (emit_srcmap) {
        rc = write_source_map(sourcemap_path, source_file, &ctx,
                              func_code_abs_offsets);
        free(func_code_abs_offsets);
        if (rc == 0) {
            printf("Source map written to %s\n", sourcemap_path);
        }
    }

    ctx_free(&ctx);
    return rc;
}

int wasm_backend_emit(ASTNode *root, const char *output_path,
                      const char *source_file, const char *sourcemap_path,
                      bool verbose)
{
    FILE *f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "[wasm] error: cannot open %s for writing\n", output_path);
        return 1;
    }
    int rc = wasm_backend_emit_fp_ex(root, f, verbose,
                                      output_path, source_file, sourcemap_path);
    fclose(f);
    return rc;
}
