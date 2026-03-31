/* lsp_server.c — nanolang Language Server Protocol server
 *
 * JSON-RPC 2.0 over stdio, Content-Length framing.
 * Features: hover, definition, completion, publishDiagnostics.
 *
 * Pipeline per document change:
 *   tokenize → parse → process_imports → type_check_module
 * Results cached for hover/definition/completion requests.
 */

#include "nanolang.h"
#include "fmt.h"
#include "cJSON.h"
/* json_diagnostics.h is NOT included here because it defines DiagnosticSeverity
 * with different member names than compiler_schema.h (included via nanolang.h).
 * Use the conflict-free accessor API declared below instead. */
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Conflict-free json_diagnostics interface (defined in json_diagnostics.c) */
extern bool g_json_output_enabled;
extern void json_diagnostics_init(void);
extern void json_diagnostics_enable(void);
extern void json_diagnostics_cleanup(void);
extern void json_error(const char *code, const char *message, const char *file,
                       int line, int column, const char *suggestion);
extern int json_diag_count(void);
extern int json_diag_severity(int idx);       /* 0=error,1=warning,2=info,3=hint */
extern const char *json_diag_code(int idx);
extern const char *json_diag_message(int idx);
extern const char *json_diag_file(int idx);
extern int json_diag_line(int idx);
extern int json_diag_column(int idx);

/* Runtime globals required by nanolang modules */
int g_argc = 0;
char **g_argv = NULL;

char g_project_root[PATH_MAX] = "";

const char *get_project_root(void) {
    return g_project_root[0] ? g_project_root : ".";
}

static void resolve_project_root(const char *argv0) {
    char exe_path[PATH_MAX];
    if (realpath(argv0, exe_path) == NULL) {
        if (getcwd(g_project_root, sizeof(g_project_root)) == NULL)
            strcpy(g_project_root, ".");
        return;
    }
    char *slash = strrchr(exe_path, '/');
    if (slash) {
        *slash = '\0';
        slash = strrchr(exe_path, '/');
        if (slash) *slash = '\0';
    }
    strncpy(g_project_root, exe_path, sizeof(g_project_root) - 1);
    g_project_root[sizeof(g_project_root) - 1] = '\0';
}

/* =========================================================================
 * Content-Length framing (LSP stdio transport)
 * ========================================================================= */

/* Read one LSP message. Returns malloc'd null-terminated JSON body, or NULL. */
static char *lsp_read_message(void) {
    char header[256];
    int content_length = -1;

    /* Read headers until blank line */
    while (fgets(header, (int)sizeof(header), stdin)) {
        size_t len = strlen(header);
        while (len > 0 && (header[len - 1] == '\r' || header[len - 1] == '\n'))
            header[--len] = '\0';
        if (len == 0)
            break; /* blank line = end of headers */
        if (strncmp(header, "Content-Length: ", 16) == 0)
            content_length = atoi(header + 16);
    }

    if (content_length <= 0)
        return NULL;

    char *buf = malloc((size_t)content_length + 1);
    if (!buf)
        return NULL;

    size_t total = 0;
    while (total < (size_t)content_length) {
        size_t n = fread(buf + total, 1, (size_t)content_length - total, stdin);
        if (n == 0) {
            free(buf);
            return NULL;
        }
        total += n;
    }
    buf[content_length] = '\0';
    return buf;
}

/* Write a JSON-RPC message with Content-Length framing to stdout. */
static void lsp_send(const char *json) {
    fprintf(stdout, "Content-Length: %d\r\n\r\n%s", (int)strlen(json), json);
    fflush(stdout);
}

/* =========================================================================
 * JSON-RPC helpers
 * ========================================================================= */

static void lsp_send_result(cJSON *id, cJSON *result) {
    cJSON *resp = cJSON_CreateObject();
    cJSON_AddStringToObject(resp, "jsonrpc", "2.0");
    if (id)
        cJSON_AddItemToObject(resp, "id", cJSON_Duplicate(id, 1));
    cJSON_AddItemToObject(resp, "result", result);
    char *str = cJSON_PrintUnformatted(resp);
    if (str) {
        lsp_send(str);
        free(str);
    }
    cJSON_Delete(resp);
}

static void lsp_send_null_result(cJSON *id) {
    lsp_send_result(id, cJSON_CreateNull());
}

static void lsp_send_error(cJSON *id, int code, const char *msg) {
    cJSON *resp = cJSON_CreateObject();
    cJSON_AddStringToObject(resp, "jsonrpc", "2.0");
    if (id)
        cJSON_AddItemToObject(resp, "id", cJSON_Duplicate(id, 1));
    cJSON *err = cJSON_CreateObject();
    cJSON_AddNumberToObject(err, "code", code);
    cJSON_AddStringToObject(err, "message", msg);
    cJSON_AddItemToObject(resp, "error", err);
    char *str = cJSON_PrintUnformatted(resp);
    if (str) {
        lsp_send(str);
        free(str);
    }
    cJSON_Delete(resp);
}

static void lsp_send_notification(const char *method, cJSON *params) {
    cJSON *notif = cJSON_CreateObject();
    cJSON_AddStringToObject(notif, "jsonrpc", "2.0");
    cJSON_AddStringToObject(notif, "method", method);
    if (params)
        cJSON_AddItemToObject(notif, "params", params);
    char *str = cJSON_PrintUnformatted(notif);
    if (str) {
        lsp_send(str);
        free(str);
    }
    cJSON_Delete(notif);
}

/* =========================================================================
 * URI / path utilities
 * ========================================================================= */

/* Convert file:// URI to filesystem path (caller must free). */
static char *uri_to_path(const char *uri) {
    if (!uri)
        return strdup("");
    if (strncmp(uri, "file://", 7) == 0)
        return strdup(uri + 7);
    return strdup(uri);
}

/* Build a temp file path in the same directory as real_path.
 * Caller must free. */
static char *make_temp_path(const char *real_path) {
    char dir[PATH_MAX];
    strncpy(dir, real_path, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    char *slash = strrchr(dir, '/');
    if (slash) {
        slash[1] = '\0';
    } else {
        dir[0] = '.';
        dir[1] = '/';
        dir[2] = '\0';
    }
    size_t needed = strlen(dir) + 32;
    char *tmp = malloc(needed);
    if (!tmp)
        return NULL;
    snprintf(tmp, needed, "%s.nanolang_lsp_tmp.nano", dir);
    return tmp;
}

/* =========================================================================
 * Type name utilities
 * ========================================================================= */

/* Return a static or stack-friendly string for a simple Type. */
static const char *type_base_name(Type t) {
    switch (t) {
        case TYPE_INT:          return "int";
        case TYPE_U8:           return "u8";
        case TYPE_FLOAT:        return "float";
        case TYPE_BOOL:         return "bool";
        case TYPE_STRING:       return "string";
        case TYPE_BSTRING:      return "bstring";
        case TYPE_VOID:         return "void";
        case TYPE_ARRAY:        return "array";
        case TYPE_STRUCT:       return "struct";
        case TYPE_ENUM:         return "enum";
        case TYPE_UNION:        return "union";
        case TYPE_GENERIC:      return "generic";
        case TYPE_LIST_INT:     return "List<int>";
        case TYPE_LIST_STRING:  return "List<string>";
        case TYPE_LIST_TOKEN:   return "List<Token>";
        case TYPE_LIST_GENERIC: return "List<T>";
        case TYPE_HASHMAP:      return "HashMap";
        case TYPE_FUNCTION:     return "fn";
        case TYPE_TUPLE:        return "tuple";
        case TYPE_OPAQUE:       return "opaque";
        case TYPE_UNKNOWN:      return "unknown";
        default:                return "unknown";
    }
}

/* Build a human-readable type label. Caller must free. */
static char *build_type_label(Type t, const char *struct_name, TypeInfo *ti) {
    if ((t == TYPE_STRUCT || t == TYPE_UNION || t == TYPE_ENUM) && struct_name)
        return strdup(struct_name);
    if (ti && ti->generic_name && ti->type_param_count > 0) {
        /* e.g. List<int> */
        char buf[256];
        snprintf(buf, sizeof(buf), "%s<", ti->generic_name);
        for (int i = 0; i < ti->type_param_count; i++) {
            if (i > 0)
                strncat(buf, ", ", sizeof(buf) - strlen(buf) - 1);
            if (ti->type_params[i]) {
                const char *pname = type_base_name(ti->type_params[i]->base_type);
                strncat(buf, pname, sizeof(buf) - strlen(buf) - 1);
            }
        }
        strncat(buf, ">", sizeof(buf) - strlen(buf) - 1);
        return strdup(buf);
    }
    if (ti && ti->opaque_type_name)
        return strdup(ti->opaque_type_name);
    return strdup(type_base_name(t));
}

/* =========================================================================
 * AST walker — find identifier at (line, col) (1-based)
 * ========================================================================= */

/* Forward declaration */
static const char *find_identifier_at(ASTNode *node, int line, int col);

static const char *check_children(ASTNode **nodes, int count, int line, int col) {
    for (int i = 0; i < count; i++) {
        const char *id = find_identifier_at(nodes[i], line, col);
        if (id)
            return id;
    }
    return NULL;
}

static const char *find_identifier_at(ASTNode *node, int line, int col) {
    if (!node)
        return NULL;

    /* Check this node first */
    if (node->type == AST_IDENTIFIER && node->line == line) {
        int id_len = (int)strlen(node->as.identifier);
        if (col >= node->column && col < node->column + id_len)
            return node->as.identifier;
    }

    /* Recurse into children */
    switch (node->type) {
        case AST_PROGRAM:
            return check_children(node->as.program.items, node->as.program.count, line, col);

        case AST_BLOCK:
            return check_children(node->as.block.statements, node->as.block.count, line, col);

        case AST_UNSAFE_BLOCK:
            return check_children(node->as.unsafe_block.statements, node->as.unsafe_block.count, line, col);

        case AST_FUNCTION: {
            const char *id = find_identifier_at(node->as.function.body, line, col);
            return id;
        }

        case AST_LET:
            return find_identifier_at(node->as.let.value, line, col);

        case AST_SET:
            return find_identifier_at(node->as.set.value, line, col);

        case AST_RETURN:
            return find_identifier_at(node->as.return_stmt.value, line, col);

        case AST_CALL: {
            /* Check function name itself */
            if (node->line == line && node->as.call.name) {
                int fn_len = (int)strlen(node->as.call.name);
                if (col >= node->column && col < node->column + fn_len)
                    return node->as.call.name;
            }
            return check_children(node->as.call.args, node->as.call.arg_count, line, col);
        }

        case AST_IF: {
            const char *id = find_identifier_at(node->as.if_stmt.condition, line, col);
            if (id) return id;
            id = find_identifier_at(node->as.if_stmt.then_branch, line, col);
            if (id) return id;
            return find_identifier_at(node->as.if_stmt.else_branch, line, col);
        }

        case AST_WHILE: {
            const char *id = find_identifier_at(node->as.while_stmt.condition, line, col);
            if (id) return id;
            return find_identifier_at(node->as.while_stmt.body, line, col);
        }

        case AST_FOR:
            return find_identifier_at(node->as.for_stmt.body, line, col);

        case AST_FIELD_ACCESS:
            return find_identifier_at(node->as.field_access.object, line, col);

        case AST_PRINT:
            return find_identifier_at(node->as.print.expr, line, col);

        case AST_ASSERT:
            return find_identifier_at(node->as.assert.condition, line, col);

        case AST_ARRAY_LITERAL:
            return check_children(node->as.array_literal.elements,
                                  node->as.array_literal.element_count, line, col);

        case AST_STRUCT_LITERAL:
            return check_children(node->as.struct_literal.field_values,
                                  node->as.struct_literal.field_count, line, col);

        case AST_MATCH: {
            const char *id = find_identifier_at(node->as.match_expr.expr, line, col);
            if (id) return id;
            return check_children(node->as.match_expr.arm_bodies,
                                  node->as.match_expr.arm_count, line, col);
        }

        case AST_TUPLE_LITERAL:
            return check_children(node->as.tuple_literal.elements,
                                  node->as.tuple_literal.element_count, line, col);

        case AST_TUPLE_INDEX:
            return find_identifier_at(node->as.tuple_index.tuple, line, col);

        case AST_TRY_OP:
            return find_identifier_at(node->as.try_op.operand, line, col);

        case AST_PAR_BLOCK:
            return check_children(node->as.par_block.bindings,
                                  node->as.par_block.count, line, col);

        case AST_COND: {
            const char *id = check_children(node->as.cond_expr.conditions,
                                            node->as.cond_expr.clause_count, line, col);
            if (id) return id;
            id = check_children(node->as.cond_expr.values,
                                node->as.cond_expr.clause_count, line, col);
            if (id) return id;
            return find_identifier_at(node->as.cond_expr.else_value, line, col);
        }

        default:
            return NULL;
    }
}

/* =========================================================================
 * Document store (single document)
 * ========================================================================= */

typedef struct {
    char *uri;          /* Document URI (malloc'd) */
    char *real_path;    /* File system path (malloc'd) */
    char *temp_path;    /* Temp file path for compilation (malloc'd) */
    char *source;       /* Current source content (malloc'd) */
    ASTNode *ast;       /* Last compiled AST (may be NULL) */
    Environment *env;   /* Last compiled environment (may be NULL) */
    Token *tokens;      /* Last token array (may be NULL) */
    int token_count;
} Document;

static Document g_doc = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0};

static void doc_free_compiled(void) {
    if (g_doc.ast) {
        free_ast(g_doc.ast);
        g_doc.ast = NULL;
    }
    if (g_doc.env) {
        free_environment(g_doc.env);
        g_doc.env = NULL;
    }
    if (g_doc.tokens) {
        free_tokens(g_doc.tokens, g_doc.token_count);
        g_doc.tokens = NULL;
        g_doc.token_count = 0;
    }
}

static void doc_set(const char *uri, const char *source) {
    if (g_doc.uri && strcmp(g_doc.uri, uri) != 0) {
        /* Different document — free everything */
        free(g_doc.uri);
        free(g_doc.real_path);
        free(g_doc.temp_path);
        free(g_doc.source);
        doc_free_compiled();
        memset(&g_doc, 0, sizeof(g_doc));
    }
    if (!g_doc.uri) {
        g_doc.uri = strdup(uri);
        g_doc.real_path = uri_to_path(uri);
        g_doc.temp_path = make_temp_path(g_doc.real_path);
    }
    free(g_doc.source);
    g_doc.source = strdup(source);
}

/* =========================================================================
 * Compilation pipeline
 * ========================================================================= */

/* Publish LSP diagnostics via the conflict-free accessor API.
 * LSP line/col are 0-based; nanolang stores 1-based. */
static void publish_diagnostics(const char *uri) {
    cJSON *params = cJSON_CreateObject();
    cJSON_AddStringToObject(params, "uri", uri);
    cJSON *diags = cJSON_CreateArray();

    int count = json_diag_count();
    for (int i = 0; i < count; i++) {
        cJSON *diag = cJSON_CreateObject();

        /* range: { start: {line,character}, end: {line,character} } */
        cJSON *range = cJSON_CreateObject();
        cJSON *start = cJSON_CreateObject();
        int raw_line = json_diag_line(i);
        int raw_col  = json_diag_column(i);
        int lsp_line = (raw_line > 0) ? raw_line - 1 : 0;
        int lsp_col  = (raw_col  > 0) ? raw_col  - 1 : 0;
        cJSON_AddNumberToObject(start, "line", lsp_line);
        cJSON_AddNumberToObject(start, "character", lsp_col);
        cJSON *end_pos = cJSON_CreateObject();
        cJSON_AddNumberToObject(end_pos, "line", lsp_line);
        cJSON_AddNumberToObject(end_pos, "character", lsp_col + 1);
        cJSON_AddItemToObject(range, "start", start);
        cJSON_AddItemToObject(range, "end", end_pos);
        cJSON_AddItemToObject(diag, "range", range);

        /* LSP severity: 1=error, 2=warning, 3=info, 4=hint
         * json_diag_severity: 0=error, 1=warning, 2=info, 3=hint */
        int raw_sev = json_diag_severity(i);
        int lsp_sev = raw_sev + 1; /* shift by 1 */
        cJSON_AddNumberToObject(diag, "severity", lsp_sev);

        const char *code = json_diag_code(i);
        if (code && code[0])
            cJSON_AddStringToObject(diag, "code", code);
        cJSON_AddStringToObject(diag, "source", "nanolang");
        const char *msg = json_diag_message(i);
        cJSON_AddStringToObject(diag, "message", msg ? msg : "");

        cJSON_AddItemToArray(diags, diag);
    }

    cJSON_AddItemToObject(params, "diagnostics", diags);
    lsp_send_notification("textDocument/publishDiagnostics", params);
}

/* Compile the current document. Returns true if type check succeeded. */
static bool doc_compile(void) {
    if (!g_doc.source || !g_doc.temp_path)
        return false;

    /* Write source to temp file for process_imports */
    FILE *f = fopen(g_doc.temp_path, "w");
    if (!f)
        return false;
    fputs(g_doc.source, f);
    fclose(f);

    /* Free previous compilation results */
    doc_free_compiled();

    /* Reset diagnostics */
    json_diagnostics_cleanup();
    json_diagnostics_init();
    json_diagnostics_enable();

    /* Phase 1: Lex */
    g_doc.tokens = tokenize(g_doc.source, &g_doc.token_count);
    if (!g_doc.tokens) {
        json_error("E000", "Lexing failed", g_doc.real_path, 1, 1, NULL);
        return false;
    }

    /* Phase 2: Parse */
    g_doc.ast = parse_program(g_doc.tokens, g_doc.token_count);
    if (!g_doc.ast) {
        json_error("E000", "Parsing failed", g_doc.real_path, 1, 1, NULL);
        return false;
    }

    /* Phase 3: Create environment and process imports */
    g_doc.env = create_environment();
    clear_module_cache();
    ModuleList *modules = create_module_list();
    bool imports_ok = process_imports(g_doc.ast, g_doc.env, modules, g_doc.temp_path);
    free_module_list(modules);
    if (!imports_ok) {
        /* Don't return false — still try to typecheck for other diagnostics */
    }

    /* Phase 4: Type check (module mode — no main() required) */
    typecheck_set_current_file(g_doc.real_path);
    bool ok = type_check_module(g_doc.ast, g_doc.env);

    /* Clean up temp file */
    remove(g_doc.temp_path);

    return ok;
}

/* =========================================================================
 * LSP request handlers
 * ========================================================================= */

static void handle_initialize(cJSON *id, cJSON *params) {
    (void)params; /* unused */

    cJSON *result = cJSON_CreateObject();
    cJSON *caps = cJSON_CreateObject();

    /* Text document sync: incremental (2) */
    cJSON *sync = cJSON_CreateObject();
    cJSON_AddNumberToObject(sync, "change", 2);
    cJSON_AddTrueToObject(sync, "openClose");
    cJSON_AddItemToObject(caps, "textDocumentSync", sync);

    /* Hover */
    cJSON_AddTrueToObject(caps, "hoverProvider");

    /* Definition */
    cJSON_AddTrueToObject(caps, "definitionProvider");

    /* Document formatting */
    cJSON_AddTrueToObject(caps, "documentFormattingProvider");

    /* Completion */
    cJSON *comp_opts = cJSON_CreateObject();
    cJSON *triggers = cJSON_CreateArray();
    cJSON_AddItemToArray(triggers, cJSON_CreateString("."));
    cJSON_AddItemToObject(comp_opts, "triggerCharacters", triggers);
    cJSON_AddItemToObject(caps, "completionProvider", comp_opts);

    cJSON_AddItemToObject(result, "capabilities", caps);

    cJSON *server_info = cJSON_CreateObject();
    cJSON_AddStringToObject(server_info, "name", "nanolang-lsp");
    cJSON_AddStringToObject(server_info, "version", "0.1.0");
    cJSON_AddItemToObject(result, "serverInfo", server_info);

    lsp_send_result(id, result);
}

static void handle_did_open(cJSON *params) {
    cJSON *td = cJSON_GetObjectItemCaseSensitive(params, "textDocument");
    if (!td) return;

    cJSON *uri_item = cJSON_GetObjectItemCaseSensitive(td, "uri");
    cJSON *text_item = cJSON_GetObjectItemCaseSensitive(td, "text");
    if (!uri_item || !text_item) return;

    const char *uri = uri_item->valuestring;
    const char *text = text_item->valuestring;
    if (!uri || !text) return;

    doc_set(uri, text);
    doc_compile();
    publish_diagnostics(uri);
}

static void handle_did_change(cJSON *params) {
    cJSON *td = cJSON_GetObjectItemCaseSensitive(params, "textDocument");
    cJSON *changes = cJSON_GetObjectItemCaseSensitive(params, "contentChanges");
    if (!td || !changes) return;

    cJSON *uri_item = cJSON_GetObjectItemCaseSensitive(td, "uri");
    if (!uri_item) return;
    const char *uri = uri_item->valuestring;
    if (!uri) return;

    /* Use the last change's text (full-document sync also sends a single change) */
    cJSON *last = NULL;
    cJSON *change = NULL;
    cJSON_ArrayForEach(change, changes) { last = change; }
    if (!last) return;

    cJSON *text_item = cJSON_GetObjectItemCaseSensitive(last, "text");
    if (!text_item || !text_item->valuestring) return;

    doc_set(uri, text_item->valuestring);
    doc_compile();
    publish_diagnostics(uri);
}

static void handle_hover(cJSON *id, cJSON *params) {
    cJSON *td = cJSON_GetObjectItemCaseSensitive(params, "textDocument");
    cJSON *pos = cJSON_GetObjectItemCaseSensitive(params, "position");
    if (!td || !pos) { lsp_send_null_result(id); return; }

    cJSON *uri_item = cJSON_GetObjectItemCaseSensitive(td, "uri");
    if (!uri_item) { lsp_send_null_result(id); return; }

    cJSON *line_item = cJSON_GetObjectItemCaseSensitive(pos, "line");
    cJSON *char_item = cJSON_GetObjectItemCaseSensitive(pos, "character");
    if (!line_item || !char_item) { lsp_send_null_result(id); return; }

    /* Convert 0-based LSP → 1-based nanolang */
    int line = (int)line_item->valuedouble + 1;
    int col  = (int)char_item->valuedouble + 1;

    if (!g_doc.ast || !g_doc.env) { lsp_send_null_result(id); return; }

    /* Find identifier under cursor */
    const char *ident = find_identifier_at(g_doc.ast, line, col);
    if (!ident) { lsp_send_null_result(id); return; }

    /* Look up symbol */
    Symbol *sym = env_get_var_visible_at(g_doc.env, ident, line, col);
    if (!sym) {
        /* Try function lookup */
        Function *fn = env_get_function(g_doc.env, ident);
        if (!fn) { lsp_send_null_result(id); return; }

        /* Build function signature label */
        char label[512];
        int off = snprintf(label, sizeof(label), "fn %s(", ident);
        for (int i = 0; i < fn->param_count && off < (int)sizeof(label) - 4; i++) {
            if (i > 0) off += snprintf(label + off, sizeof(label) - (size_t)off, ", ");
            char *tname = build_type_label(fn->params[i].type,
                                           fn->params[i].struct_type_name,
                                           fn->params[i].type_info);
            off += snprintf(label + off, sizeof(label) - (size_t)off,
                            "%s: %s", fn->params[i].name ? fn->params[i].name : "_", tname);
            free(tname);
        }
        char *retname = build_type_label(fn->return_type, fn->return_struct_type_name,
                                         fn->return_type_info);
        snprintf(label + off, sizeof(label) - (size_t)off, ") -> %s", retname);
        free(retname);

        cJSON *result = cJSON_CreateObject();
        cJSON *contents = cJSON_CreateObject();
        cJSON_AddStringToObject(contents, "kind", "markdown");
        cJSON_AddStringToObject(contents, "value", label);
        cJSON_AddItemToObject(result, "contents", contents);
        lsp_send_result(id, result);
        return;
    }

    /* Symbol found — build type string */
    char *tname = build_type_label(sym->type, sym->struct_type_name, sym->type_info);
    char label[256];
    const char *mut_str = sym->is_mut ? "mut " : "";
    snprintf(label, sizeof(label), "%s%s: %s", mut_str, ident, tname);
    free(tname);

    cJSON *result = cJSON_CreateObject();
    cJSON *contents = cJSON_CreateObject();
    cJSON_AddStringToObject(contents, "kind", "markdown");
    cJSON_AddStringToObject(contents, "value", label);
    cJSON_AddItemToObject(result, "contents", contents);
    lsp_send_result(id, result);
}

static void handle_definition(cJSON *id, cJSON *params) {
    cJSON *td = cJSON_GetObjectItemCaseSensitive(params, "textDocument");
    cJSON *pos = cJSON_GetObjectItemCaseSensitive(params, "position");
    if (!td || !pos) { lsp_send_null_result(id); return; }

    cJSON *uri_item = cJSON_GetObjectItemCaseSensitive(td, "uri");
    if (!uri_item) { lsp_send_null_result(id); return; }

    cJSON *line_item = cJSON_GetObjectItemCaseSensitive(pos, "line");
    cJSON *char_item = cJSON_GetObjectItemCaseSensitive(pos, "character");
    if (!line_item || !char_item) { lsp_send_null_result(id); return; }

    int line = (int)line_item->valuedouble + 1;
    int col  = (int)char_item->valuedouble + 1;

    if (!g_doc.ast || !g_doc.env) { lsp_send_null_result(id); return; }

    const char *ident = find_identifier_at(g_doc.ast, line, col);
    if (!ident) { lsp_send_null_result(id); return; }

    Symbol *sym = env_get_var_visible_at(g_doc.env, ident, line, col);
    if (!sym || sym->def_line <= 0) {
        /* Try function */
        Function *fn = env_get_function(g_doc.env, ident);
        if (!fn || !fn->body) { lsp_send_null_result(id); return; }
        /* Use function body's position */
        cJSON *result = cJSON_CreateObject();
        const char *uri = uri_item->valuestring;
        cJSON_AddStringToObject(result, "uri", uri);
        cJSON *range = cJSON_CreateObject();
        cJSON *fn_start = cJSON_CreateObject();
        int fline = fn->body->line > 0 ? fn->body->line - 1 : 0;
        cJSON_AddNumberToObject(fn_start, "line", fline);
        cJSON_AddNumberToObject(fn_start, "character", 0);
        cJSON *fn_end = cJSON_CreateObject();
        cJSON_AddNumberToObject(fn_end, "line", fline);
        cJSON_AddNumberToObject(fn_end, "character", 0);
        cJSON_AddItemToObject(range, "start", fn_start);
        cJSON_AddItemToObject(range, "end", fn_end);
        cJSON_AddItemToObject(result, "range", range);
        lsp_send_result(id, result);
        return;
    }

    /* Return definition location */
    int def_lsp_line = sym->def_line - 1;
    int def_lsp_col  = sym->def_column > 0 ? sym->def_column - 1 : 0;

    const char *uri = g_doc.uri ? g_doc.uri : uri_item->valuestring;
    cJSON *result = cJSON_CreateObject();
    cJSON_AddStringToObject(result, "uri", uri);
    cJSON *range = cJSON_CreateObject();
    cJSON *rstart = cJSON_CreateObject();
    cJSON_AddNumberToObject(rstart, "line", def_lsp_line);
    cJSON_AddNumberToObject(rstart, "character", def_lsp_col);
    cJSON *rend = cJSON_CreateObject();
    cJSON_AddNumberToObject(rend, "line", def_lsp_line);
    cJSON_AddNumberToObject(rend, "character", def_lsp_col + (int)strlen(ident));
    cJSON_AddItemToObject(range, "start", rstart);
    cJSON_AddItemToObject(range, "end", rend);
    cJSON_AddItemToObject(result, "range", range);
    lsp_send_result(id, result);
}

/*
 * textDocument/formatting — reformat the open document using nano-fmt.
 *
 * Returns a list of TextEdit objects replacing the entire document content
 * with the formatted version.
 */
static void handle_formatting(cJSON *id, cJSON *params) {
    if (!g_doc.source || !g_doc.source[0]) {
        lsp_send_result(id, cJSON_CreateArray());
        return;
    }

    /* Determine indent size from formattingOptions */
    int tab_size = 4;
    if (params) {
        cJSON *fo = cJSON_GetObjectItemCaseSensitive(params, "options");
        if (fo) {
            cJSON *ts = cJSON_GetObjectItemCaseSensitive(fo, "tabSize");
            if (ts && cJSON_IsNumber(ts)) tab_size = ts->valueint;
        }
    }

    FmtOptions fmt_opts = {
        .indent_size    = tab_size,
        .write_in_place = false,
        .check_only     = false,
        .verbose        = false,
    };

    char *formatted = fmt_source(g_doc.source, g_doc.real_path, &fmt_opts);
    if (!formatted) {
        lsp_send_result(id, cJSON_CreateArray());
        return;
    }

    /* Count lines in original source to build a full-document range */
    int line_count = 0;
    for (const char *p = g_doc.source; *p; p++) {
        if (*p == '\n') line_count++;
    }
    /* Last line's last character */
    const char *last_line_start = strrchr(g_doc.source, '\n');
    int last_col = last_line_start
                 ? (int)strlen(last_line_start + 1)
                 : (int)strlen(g_doc.source);

    /* Build TextEdit: replace entire file */
    cJSON *edit  = cJSON_CreateObject();
    cJSON *range = cJSON_CreateObject();
    cJSON *start = cJSON_CreateObject();
    cJSON *end   = cJSON_CreateObject();
    cJSON_AddNumberToObject(start, "line",      0);
    cJSON_AddNumberToObject(start, "character", 0);
    cJSON_AddNumberToObject(end,   "line",      line_count);
    cJSON_AddNumberToObject(end,   "character", last_col);
    cJSON_AddItemToObject(range, "start", start);
    cJSON_AddItemToObject(range, "end",   end);
    cJSON_AddItemToObject(edit,  "range", range);
    cJSON_AddStringToObject(edit, "newText", formatted);
    free(formatted);

    cJSON *edits = cJSON_CreateArray();
    cJSON_AddItemToArray(edits, edit);
    lsp_send_result(id, edits);
}

static void handle_completion(cJSON *id, cJSON *params) {
    cJSON *td = cJSON_GetObjectItemCaseSensitive(params, "textDocument");
    cJSON *pos = cJSON_GetObjectItemCaseSensitive(params, "position");
    if (!td || !pos) { lsp_send_null_result(id); return; }

    cJSON *line_item = cJSON_GetObjectItemCaseSensitive(pos, "line");
    cJSON *char_item = cJSON_GetObjectItemCaseSensitive(pos, "character");
    int line = line_item ? (int)line_item->valuedouble + 1 : 0;
    int col  = char_item ? (int)char_item->valuedouble + 1 : 0;
    (void)line; (void)col; /* used for future scope filtering */

    if (!g_doc.env) { lsp_send_null_result(id); return; }

    cJSON *items = cJSON_CreateArray();

    /* Add symbols (variables) */
    for (int i = 0; i < g_doc.env->symbol_count; i++) {
        Symbol *sym = &g_doc.env->symbols[i];
        if (!sym->name) continue;

        cJSON *item = cJSON_CreateObject();
        cJSON_AddStringToObject(item, "label", sym->name);
        cJSON_AddNumberToObject(item, "kind", 6); /* 6 = Variable */

        char *tname = build_type_label(sym->type, sym->struct_type_name, sym->type_info);
        char detail[128];
        snprintf(detail, sizeof(detail), "%s: %s", sym->name, tname);
        free(tname);
        cJSON_AddStringToObject(item, "detail", detail);

        cJSON_AddItemToArray(items, item);
    }

    /* Add functions */
    for (int i = 0; i < g_doc.env->function_count; i++) {
        Function *fn = &g_doc.env->functions[i];
        if (!fn->name) continue;

        cJSON *item = cJSON_CreateObject();
        cJSON_AddStringToObject(item, "label", fn->name);
        cJSON_AddNumberToObject(item, "kind", 3); /* 3 = Function */

        char detail[256];
        int off = snprintf(detail, sizeof(detail), "fn %s(", fn->name);
        for (int j = 0; j < fn->param_count && off < (int)sizeof(detail) - 4; j++) {
            if (j > 0) off += snprintf(detail + off, sizeof(detail) - (size_t)off, ", ");
            char *tname = build_type_label(fn->params[j].type,
                                           fn->params[j].struct_type_name,
                                           fn->params[j].type_info);
            off += snprintf(detail + off, sizeof(detail) - (size_t)off, "%s", tname);
            free(tname);
        }
        char *retname = build_type_label(fn->return_type, fn->return_struct_type_name,
                                          fn->return_type_info);
        snprintf(detail + off, sizeof(detail) - (size_t)off, ") -> %s", retname);
        free(retname);
        cJSON_AddStringToObject(item, "detail", detail);

        cJSON_AddItemToArray(items, item);
    }

    /* Add struct names */
    for (int i = 0; i < g_doc.env->struct_count; i++) {
        StructDef *sd = &g_doc.env->structs[i];
        if (!sd->name) continue;
        cJSON *item = cJSON_CreateObject();
        cJSON_AddStringToObject(item, "label", sd->name);
        cJSON_AddNumberToObject(item, "kind", 22); /* 22 = Struct */
        cJSON_AddItemToArray(items, item);
    }

    /* Add enum names */
    for (int i = 0; i < g_doc.env->enum_count; i++) {
        EnumDef *ed = &g_doc.env->enums[i];
        if (!ed->name) continue;
        cJSON *item = cJSON_CreateObject();
        cJSON_AddStringToObject(item, "label", ed->name);
        cJSON_AddNumberToObject(item, "kind", 13); /* 13 = Enum */
        cJSON_AddItemToArray(items, item);
    }

    lsp_send_result(id, items);
}

/* =========================================================================
 * Main message loop
 * ========================================================================= */

static void dispatch(cJSON *msg) {
    cJSON *method_item = cJSON_GetObjectItemCaseSensitive(msg, "method");
    cJSON *id = cJSON_GetObjectItemCaseSensitive(msg, "id");
    cJSON *params = cJSON_GetObjectItemCaseSensitive(msg, "params");
    if (!method_item || !method_item->valuestring) return;

    const char *method = method_item->valuestring;

    if (strcmp(method, "initialize") == 0) {
        handle_initialize(id, params);
    } else if (strcmp(method, "initialized") == 0) {
        /* notification — no response needed */
    } else if (strcmp(method, "shutdown") == 0) {
        lsp_send_result(id, cJSON_CreateNull());
    } else if (strcmp(method, "exit") == 0) {
        exit(0);
    } else if (strcmp(method, "textDocument/didOpen") == 0) {
        if (params) handle_did_open(params);
    } else if (strcmp(method, "textDocument/didChange") == 0) {
        if (params) handle_did_change(params);
    } else if (strcmp(method, "textDocument/didClose") == 0) {
        /* nothing to do */
    } else if (strcmp(method, "textDocument/hover") == 0) {
        if (params) handle_hover(id, params);
        else lsp_send_null_result(id);
    } else if (strcmp(method, "textDocument/definition") == 0) {
        if (params) handle_definition(id, params);
        else lsp_send_null_result(id);
    } else if (strcmp(method, "textDocument/completion") == 0) {
        if (params) handle_completion(id, params);
        else lsp_send_null_result(id);
    } else if (strcmp(method, "textDocument/formatting") == 0) {
        handle_formatting(id, params);
    } else if (id) {
        /* Unknown method with id — send method-not-found error */
        lsp_send_error(id, -32601, "Method not found");
    }
    /* Unknown notifications are silently ignored */
}

int main(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
    resolve_project_root(argv[0]);

    /* Use binary I/O to avoid CR/LF translation issues on some platforms */
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    json_diagnostics_init();

    for (;;) {
        char *body = lsp_read_message();
        if (!body) break; /* EOF */

        cJSON *msg = cJSON_Parse(body);
        free(body);
        if (!msg) continue;

        dispatch(msg);
        cJSON_Delete(msg);
    }

    /* Cleanup */
    doc_free_compiled();
    free(g_doc.uri);
    free(g_doc.real_path);
    free(g_doc.temp_path);
    free(g_doc.source);
    json_diagnostics_cleanup();
    return 0;
}
