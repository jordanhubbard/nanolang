#include "reflection.h"
#include "nanolang.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

/* Helper: Get type name as string */
static const char *type_name(Type t) {
    switch (t) {
        case TYPE_INT: return "int";
        case TYPE_FLOAT: return "float";
        case TYPE_STRING: return "string";
        case TYPE_BOOL: return "bool";
        case TYPE_VOID: return "void";
        case TYPE_ARRAY: return "array";
        case TYPE_STRUCT: return "struct";
        case TYPE_ENUM: return "enum";
        case TYPE_UNION: return "union";
        case TYPE_OPAQUE: return "opaque";
        case TYPE_TUPLE: return "tuple";
        default: return "unknown";
    }
}

/* Emit function signature as JSON */
static void emit_function_json(FILE *out, Function *func, bool *first) {
    if (!*first) fprintf(out, ",\n");
    *first = false;
    
    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"function\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, func->name);
    fprintf(out, ",\n");
    
    /* Build signature string */
    fprintf(out, "      \"signature\": \"");
    if (func->is_extern) fprintf(out, "extern ");
    fprintf(out, "fn %s(", func->name ? func->name : "");
    if (func->params) {
        for (int i = 0; i < func->param_count; i++) {
            if (i > 0) fprintf(out, ", ");
            fprintf(out, "%s: %s", 
                    func->params[i].name ? func->params[i].name : "", 
                    type_name(func->params[i].type));
            /* Add struct/enum type name if available */
            if (func->params[i].type == TYPE_STRUCT && func->params[i].struct_type_name) {
                fprintf(out, "<%s>", func->params[i].struct_type_name);
            }
        }
    }
    fprintf(out, ") -> %s", type_name(func->return_type));
    if (func->return_type == TYPE_STRUCT && func->return_struct_type_name) {
        fprintf(out, "<%s>", func->return_struct_type_name);
    }
    fprintf(out, "\",\n");
    
    /* Parameters array */
    fprintf(out, "      \"params\": [");
    if (func->params) {
        for (int i = 0; i < func->param_count; i++) {
            if (i > 0) fprintf(out, ", ");
            fprintf(out, "\n        {");
            fprintf(out, "\"name\": ");
            json_escape_string(out, func->params[i].name);
            fprintf(out, ", \"type\": \"%s\"", type_name(func->params[i].type));
            if (func->params[i].type == TYPE_STRUCT && func->params[i].struct_type_name) {
                fprintf(out, ", \"struct_name\": ");
                json_escape_string(out, func->params[i].struct_type_name);
            }
            fprintf(out, "}");
        }
        if (func->param_count > 0) fprintf(out, "\n      ");
    }
    fprintf(out, "],\n");
    
    /* Return type */
    fprintf(out, "      \"return_type\": \"%s\"", type_name(func->return_type));
    if (func->return_type == TYPE_STRUCT && func->return_struct_type_name) {
        fprintf(out, ",\n      \"return_struct_name\": ");
        json_escape_string(out, func->return_struct_type_name);
    }
    fprintf(out, ",\n");
    
    /* Flags */
    fprintf(out, "      \"is_extern\": %s,\n", func->is_extern ? "true" : "false");
    fprintf(out, "      \"is_public\": %s\n", func->is_pub ? "true" : "false");
    
    fprintf(out, "    }");
}

/* Emit struct definition as JSON */
static void emit_struct_json(FILE *out, StructDef *s, bool *first) {
    if (!*first) fprintf(out, ",\n");
    *first = false;
    
    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"struct\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, s->name);
    fprintf(out, ",\n");
    
    /* Fields array */
    fprintf(out, "      \"fields\": [");
    for (int i = 0; i < s->field_count; i++) {
        if (i > 0) fprintf(out, ", ");
        fprintf(out, "\n        {");
        fprintf(out, "\"name\": ");
        json_escape_string(out, s->field_names[i]);
        fprintf(out, ", \"type\": \"%s\"", type_name(s->field_types[i]));
        if (s->field_types[i] == TYPE_STRUCT && s->field_type_names && s->field_type_names[i]) {
            fprintf(out, ", \"struct_name\": ");
            json_escape_string(out, s->field_type_names[i]);
        }
        fprintf(out, "}");
    }
    if (s->field_count > 0) fprintf(out, "\n      ");
    fprintf(out, "],\n");
    
    /* Flags */
    fprintf(out, "      \"is_resource\": %s,\n", s->is_resource ? "true" : "false");
    fprintf(out, "      \"is_public\": %s\n", s->is_pub ? "true" : "false");
    
    fprintf(out, "    }");
}

/* Emit enum definition as JSON */
static void emit_enum_json(FILE *out, EnumDef *e, bool *first) {
    if (!*first) fprintf(out, ",\n");
    *first = false;
    
    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"enum\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, e->name);
    fprintf(out, ",\n");
    
    /* Variants array */
    fprintf(out, "      \"variants\": [");
    for (int i = 0; i < e->variant_count; i++) {
        if (i > 0) fprintf(out, ", ");
        fprintf(out, "\n        ");
        json_escape_string(out, e->variant_names[i]);
    }
    if (e->variant_count > 0) fprintf(out, "\n      ");
    fprintf(out, "],\n");
    
    /* Flags */
    fprintf(out, "      \"is_public\": %s\n", e->is_pub ? "true" : "false");
    
    fprintf(out, "    }");
}

/* Emit union definition as JSON */
static void emit_union_json(FILE *out, UnionDef *u, bool *first) {
    if (!*first) fprintf(out, ",\n");
    *first = false;
    
    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"union\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, u->name);
    fprintf(out, ",\n");
    
    /* Variants array */
    fprintf(out, "      \"variants\": [");
    for (int i = 0; i < u->variant_count; i++) {
        if (i > 0) fprintf(out, ", ");
        fprintf(out, "\n        {");
        fprintf(out, "\"name\": ");
        json_escape_string(out, u->variant_names[i]);
        
        /* Include fields if this variant has any */
        if (u->variant_field_counts && u->variant_field_counts[i] > 0) {
            fprintf(out, ", \"fields\": [");
            for (int j = 0; j < u->variant_field_counts[i]; j++) {
                if (j > 0) fprintf(out, ", ");
                fprintf(out, "{\"name\": ");
                json_escape_string(out, u->variant_field_names[i][j]);
                fprintf(out, ", \"type\": \"%s\"", type_name(u->variant_field_types[i][j]));
                if (u->variant_field_types[i][j] == TYPE_STRUCT && 
                    u->variant_field_type_names && 
                    u->variant_field_type_names[i] && 
                    u->variant_field_type_names[i][j]) {
                    fprintf(out, ", \"struct_name\": ");
                    json_escape_string(out, u->variant_field_type_names[i][j]);
                }
                fprintf(out, "}");
            }
            fprintf(out, "]");
        }
        fprintf(out, "}");
    }
    if (u->variant_count > 0) fprintf(out, "\n      ");
    fprintf(out, "],\n");
    
    /* Flags */
    fprintf(out, "      \"is_generic\": %s,\n", (u->generic_param_count > 0) ? "true" : "false");
    fprintf(out, "      \"is_public\": %s\n", u->is_pub ? "true" : "false");
    
    fprintf(out, "    }");
}

/* Emit opaque type definition as JSON */
static void emit_opaque_json(FILE *out, OpaqueTypeDef *o, bool *first) {
    if (!*first) fprintf(out, ",\n");
    *first = false;
    
    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"opaque\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, o->name);
    fprintf(out, "\n");
    fprintf(out, "    }");
}

/* Emit constant/variable as JSON */
static void emit_constant_json(FILE *out, Symbol *sym, bool *first) {
    /* Only emit constants (non-mutable top-level lets) */
    if (sym->is_mut) return;
    
    if (!*first) fprintf(out, ",\n");
    *first = false;
    
    fprintf(out, "    {\n");
    fprintf(out, "      \"kind\": \"constant\",\n");
    fprintf(out, "      \"name\": ");
    json_escape_string(out, sym->name);
    fprintf(out, ",\n");
    fprintf(out, "      \"type\": \"%s\"", type_name(sym->type));
    
    /* Add value if available (for simple types) */
    if (sym->value.type == VAL_INT) {
        fprintf(out, ",\n      \"value\": %lld", sym->value.as.int_val);
    } else if (sym->value.type == VAL_FLOAT) {
        fprintf(out, ",\n      \"value\": %f", sym->value.as.float_val);
    } else if (sym->value.type == VAL_BOOL) {
        fprintf(out, ",\n      \"value\": %s", sym->value.as.bool_val ? "true" : "false");
    } else if (sym->value.type == VAL_STRING && sym->value.as.string_val) {
        fprintf(out, ",\n      \"value\": ");
        json_escape_string(out, sym->value.as.string_val);
    }
    
    fprintf(out, "\n    }");
}

/* Main reflection function - emit module exports as JSON */
bool emit_module_reflection(const char *output_path, Environment *env, const char *module_name) {
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
    
    /* Emit all functions */
    for (int i = 0; i < env->function_count; i++) {
        Function *func = &env->functions[i];
        /* Skip internal/builtin functions */
        if (!func->name || func->name[0] == '_') continue;
        emit_function_json(out, func, &first);
    }
    
    /* Emit all structs */
    for (int i = 0; i < env->struct_count; i++) {
        StructDef *s = &env->structs[i];
        if (!s->name || s->name[0] == '_') continue;
        emit_struct_json(out, s, &first);
    }
    
    /* Emit all enums */
    for (int i = 0; i < env->enum_count; i++) {
        EnumDef *e = &env->enums[i];
        if (!e->name || e->name[0] == '_') continue;
        emit_enum_json(out, e, &first);
    }
    
    /* Emit all unions */
    for (int i = 0; i < env->union_count; i++) {
        UnionDef *u = &env->unions[i];
        if (!u->name || u->name[0] == '_') continue;
        emit_union_json(out, u, &first);
    }
    
    /* Emit all opaque types */
    for (int i = 0; i < env->opaque_type_count; i++) {
        OpaqueTypeDef *o = &env->opaque_types[i];
        if (!o->name || o->name[0] == '_') continue;
        emit_opaque_json(out, o, &first);
    }
    
    /* Emit all constants */
    for (int i = 0; i < env->symbol_count; i++) {
        Symbol *sym = &env->symbols[i];
        if (!sym->name || sym->name[0] == '_') continue;
        if (sym->from_c_header) continue;  /* Skip C header constants */
        emit_constant_json(out, sym, &first);
    }
    
    fprintf(out, "\n  ]\n");
    fprintf(out, "}\n");
    
    fclose(out);
    return true;
}
