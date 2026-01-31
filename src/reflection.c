#include "reflection.h"
#include "nanolang.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

/* Helper: JSON escape string */
static void json_escape_string(FILE *out, const char *s) {
    if (!s) {
        fprintf(out, "null");
        return;
    }
    
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '\\': fputs("\\\\", out); break;
            case '"': fputs("\\\"", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (c < 0x20) fprintf(out, "\\u%04x", (unsigned int)c);
                else fputc((int)c, out);
        }
    }
    fputc('"', out);
}

static void buf_appendf(char *buf, size_t cap, size_t *off, const char *fmt, ...) {
    if (!buf || cap == 0 || !off || *off >= cap) return;

    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf + *off, cap - *off, fmt, args);
    va_end(args);

    if (n <= 0) return;
    size_t nn = (size_t)n;
    if (nn >= cap - *off) {
        *off = cap - 1;
        buf[cap - 1] = '\0';
        return;
    }
    *off += nn;
}

static void append_typeinfo_to_buf(char *buf, size_t cap, size_t *off, const TypeInfo *ti);

static void append_type_to_buf(char *buf, size_t cap, size_t *off, Type t, const char *struct_name, Type element_type, const TypeInfo *type_info) {
    switch (t) {
        case TYPE_INT: buf_appendf(buf, cap, off, "int"); return;
        case TYPE_U8: buf_appendf(buf, cap, off, "u8"); return;
        case TYPE_FLOAT: buf_appendf(buf, cap, off, "float"); return;
        case TYPE_BOOL: buf_appendf(buf, cap, off, "bool"); return;
        case TYPE_STRING: buf_appendf(buf, cap, off, "string"); return;
        case TYPE_VOID: buf_appendf(buf, cap, off, "void"); return;

        case TYPE_ARRAY: {
            buf_appendf(buf, cap, off, "array<");
            if (type_info && type_info->element_type) {
                append_typeinfo_to_buf(buf, cap, off, type_info->element_type);
            } else {
                append_type_to_buf(buf, cap, off, element_type, NULL, TYPE_UNKNOWN, NULL);
            }
            buf_appendf(buf, cap, off, ">");
            return;
        }

        case TYPE_HASHMAP:
            if (type_info) {
                append_typeinfo_to_buf(buf, cap, off, type_info);
            } else {
                buf_appendf(buf, cap, off, "HashMap");
            }
            return;

        case TYPE_STRUCT:
        case TYPE_ENUM:
        case TYPE_UNION:
        case TYPE_OPAQUE:
            if (struct_name && struct_name[0] != '\0') {
                buf_appendf(buf, cap, off, "%s", struct_name);
            } else if (type_info && type_info->generic_name) {
                buf_appendf(buf, cap, off, "%s", type_info->generic_name);
            } else {
                buf_appendf(buf, cap, off, "unknown");
            }
            return;

        default:
            buf_appendf(buf, cap, off, "unknown");
            return;
    }
}

static void append_typeinfo_to_buf(char *buf, size_t cap, size_t *off, const TypeInfo *ti) {
    if (!ti) {
        buf_appendf(buf, cap, off, "unknown");
        return;
    }

    if (ti->base_type == TYPE_HASHMAP && ti->generic_name && strcmp(ti->generic_name, "HashMap") == 0 && ti->type_param_count == 2) {
        buf_appendf(buf, cap, off, "HashMap<");
        append_typeinfo_to_buf(buf, cap, off, ti->type_params[0]);
        buf_appendf(buf, cap, off, ",");
        append_typeinfo_to_buf(buf, cap, off, ti->type_params[1]);
        buf_appendf(buf, cap, off, ">");
        return;
    }

    if (ti->base_type == TYPE_ARRAY && ti->element_type) {
        buf_appendf(buf, cap, off, "array<");
        append_typeinfo_to_buf(buf, cap, off, ti->element_type);
        buf_appendf(buf, cap, off, ">");
        return;
    }

    if ((ti->base_type == TYPE_STRUCT || ti->base_type == TYPE_UNION) && ti->generic_name) {
        buf_appendf(buf, cap, off, "%s", ti->generic_name);
        if (ti->type_param_count > 0 && ti->type_params) {
            buf_appendf(buf, cap, off, "<");
            for (int i = 0; i < ti->type_param_count; i++) {
                if (i > 0) buf_appendf(buf, cap, off, ",");
                append_typeinfo_to_buf(buf, cap, off, ti->type_params[i]);
            }
            buf_appendf(buf, cap, off, ">");
        }
        return;
    }

    append_type_to_buf(buf, cap, off, ti->base_type, ti->generic_name, TYPE_UNKNOWN, NULL);
}

static void emit_function_json(FILE *out, ASTNode *fn, bool *first) {
    if (!out || !fn || fn->type != AST_FUNCTION) return;
    if (!*first) fprintf(out, ",\n");
    *first = false;

    const char *name = fn->as.function.name ? fn->as.function.name : "";

    char sig[4096];
    size_t off = 0;
    sig[0] = '\0';

    if (fn->as.function.is_extern) {
        buf_appendf(sig, sizeof(sig), &off, "extern ");
    }
    buf_appendf(sig, sizeof(sig), &off, "fn %s(", name);
    for (int i = 0; i < fn->as.function.param_count; i++) {
        Parameter *p = &fn->as.function.params[i];
        if (i > 0) buf_appendf(sig, sizeof(sig), &off, ", ");
        buf_appendf(sig, sizeof(sig), &off, "%s: ", p->name ? p->name : "");
        append_type_to_buf(sig, sizeof(sig), &off, p->type, p->struct_type_name, p->element_type, p->type_info);
    }
    buf_appendf(sig, sizeof(sig), &off, ") -> ");
    append_type_to_buf(sig, sizeof(sig), &off,
                       fn->as.function.return_type,
                       fn->as.function.return_struct_type_name,
                       fn->as.function.return_element_type,
                       fn->as.function.return_type_info);

    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"function\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, name);
    fprintf(out, ",\n");
    fprintf(out, "      \"signature\": ");
    json_escape_string(out, sig);
    fprintf(out, ",\n");

    fprintf(out, "      \"params\": [");
    for (int i = 0; i < fn->as.function.param_count; i++) {
        Parameter *p = &fn->as.function.params[i];
        if (i > 0) fprintf(out, ", ");
        fprintf(out, "\n        {");
        fprintf(out, "\"name\": ");
        json_escape_string(out, p->name);
        fprintf(out, ", \"type\": ");
        char tbuf[512];
        size_t toff = 0;
        tbuf[0] = '\0';
        append_type_to_buf(tbuf, sizeof(tbuf), &toff, p->type, p->struct_type_name, p->element_type, p->type_info);
        json_escape_string(out, tbuf);
        fprintf(out, "}");
    }
    if (fn->as.function.param_count > 0) fprintf(out, "\n      ");
    fprintf(out, "],\n");

    fprintf(out, "      \"return_type\": ");
    char rbuf[512];
    size_t roff = 0;
    rbuf[0] = '\0';
    append_type_to_buf(rbuf, sizeof(rbuf), &roff,
                       fn->as.function.return_type,
                       fn->as.function.return_struct_type_name,
                       fn->as.function.return_element_type,
                       fn->as.function.return_type_info);
    json_escape_string(out, rbuf);
    fprintf(out, ",\n");

    fprintf(out, "      \"is_extern\": %s,\n", fn->as.function.is_extern ? "true" : "false");
    fprintf(out, "      \"is_public\": %s\n", fn->as.function.is_pub ? "true" : "false");
    fprintf(out, "    }");
}

static void emit_opaque_json(FILE *out, ASTNode *node, bool *first) {
    if (!out || !node || node->type != AST_OPAQUE_TYPE) return;
    if (!*first) fprintf(out, ",\n");
    *first = false;

    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"opaque\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, node->as.opaque_type.name);
    fprintf(out, "\n");
    fprintf(out, "    }");
}

static void emit_constant_json(FILE *out, ASTNode *node, bool *first) {
    if (!out || !node || node->type != AST_LET) return;
    if (node->as.let.is_mut) return;
    if (!node->as.let.name) return;

    if (!*first) fprintf(out, ",\n");
    *first = false;

    char tbuf[512];
    size_t toff = 0;
    tbuf[0] = '\0';
    append_type_to_buf(tbuf, sizeof(tbuf), &toff, node->as.let.var_type, node->as.let.type_name, node->as.let.element_type, node->as.let.type_info);

    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"constant\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, node->as.let.name);
    fprintf(out, ",\n");
    fprintf(out, "      \"type\": ");
    json_escape_string(out, tbuf);

    if (node->as.let.value) {
        ASTNode *v = node->as.let.value;
        if (v->type == AST_NUMBER) {
            fprintf(out, ",\n      \"value\": %lld", v->as.number);
        } else if (v->type == AST_FLOAT) {
            fprintf(out, ",\n      \"value\": %f", v->as.float_val);
        } else if (v->type == AST_BOOL) {
            fprintf(out, ",\n      \"value\": %s", v->as.bool_val ? "true" : "false");
        } else if (v->type == AST_STRING) {
            fprintf(out, ",\n      \"value\": ");
            json_escape_string(out, v->as.string_val);
        }
    }

    fprintf(out, "\n    }");
}

/* Main reflection function - emit module exports as JSON */
bool emit_module_reflection(const char *output_path, ASTNode *program, Environment *env, const char *module_name) {
    FILE *out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "Error: Could not open output file for reflection: %s\n", output_path);
        return false;
    }
    
    fprintf(out, "{\n");
    fprintf(out, "  \"module\": ");
    json_escape_string(out, module_name);
    fprintf(out, ",\n");
    fprintf(out, "  \"exports\": [\n");
    
    bool first = true;

    (void)env;
    if (program && program->type == AST_PROGRAM) {
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
            if (!item) continue;

            switch (item->type) {
                case AST_FUNCTION:
                    if (item->as.function.name && item->as.function.name[0] != '_') {
                        emit_function_json(out, item, &first);
                    }
                    break;
                case AST_OPAQUE_TYPE:
                    if (item->as.opaque_type.name && item->as.opaque_type.name[0] != '_') {
                        emit_opaque_json(out, item, &first);
                    }
                    break;
                case AST_LET:
                    if (item->as.let.name && item->as.let.name[0] != '_') {
                        emit_constant_json(out, item, &first);
                    }
                    break;
                default:
                    break;
            }
        }
    }
    
    fprintf(out, "\n  ]\n");
    fprintf(out, "}\n");
    
    fclose(out);
    return true;
}
