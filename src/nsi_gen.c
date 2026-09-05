#include "nsi_gen.h"

#include <string.h>

static void ident_copy(char *dst, size_t n, const char *id, char sep) {
    size_t i = 0;
    size_t o = 0;
    if (!dst || n == 0) return;
    if (id && id[0] && !((id[0] >= 'A' && id[0] <= 'Z') || (id[0] >= 'a' && id[0] <= 'z') || id[0] == '_')) {
        if (o + 1 < n) dst[o++] = 'n';
    }
    for (i = 0; id && id[i] && o + 1 < n; i++) {
        unsigned char c = (unsigned char)id[i];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
            dst[o++] = (char)c;
        else
            dst[o++] = sep;
    }
    dst[o] = '\0';
}

static int is_in_param(const NlNsiParam *p) {
    return p && (p->direction == NL_NSI_DIR_IN || p->direction == NL_NSI_DIR_INOUT);
}

static const char *nano_type(const NlNsi *nsi, const char *type_id) {
    size_t i;
    if (!type_id) return NULL;
    if (strcmp(type_id, "nsi:core/string") == 0) return "string";
    if (strcmp(type_id, "nsi:core/bytes") == 0) return "string";
    if (strcmp(type_id, "nsi:core/int") == 0) return "int";
    if (strcmp(type_id, "nsi:core/bool") == 0) return "bool";
    if (strcmp(type_id, "nsi:core/float") == 0) return "float";
    if (strcmp(type_id, "nsi:core/unit") == 0) return "int";
    if (!nsi) return NULL;
    for (i = 0; i < nsi->type_count; i++) {
        if (!nsi->types[i].id || strcmp(nsi->types[i].id, type_id) != 0) continue;
        if (nsi->types[i].kind == NL_NSI_TYPE_VARIANT) return "int";
        if (nsi->types[i].kind == NL_NSI_TYPE_STRING) return "string";
        if (nsi->types[i].kind == NL_NSI_TYPE_BINARY) return "string";
        if (nsi->types[i].kind == NL_NSI_TYPE_RESOURCE) return "int";
        return NULL;
    }
    return NULL;
}

static const char *shadow_lit(const char *nano_ty) {
    if (!nano_ty) return "0";
    if (strcmp(nano_ty, "string") == 0) return "\"\"";
    if (strcmp(nano_ty, "bool") == 0) return "false";
    if (strcmp(nano_ty, "float") == 0) return "0.0";
    return "0";
}

static int emit_in_params_nano(const NlNsi *nsi, const NlNsiMethod *m, FILE *out) {
    size_t j;
    int first = 1;
    for (j = 0; j < m->param_count; j++) {
        const NlNsiParam *p = &m->params[j];
        const char *ty;
        if (!is_in_param(p)) continue;
        ty = nano_type(nsi, p->type_id);
        if (!ty) return -1;
        if (!first) fprintf(out, ", ");
        first = 0;
        fprintf(out, "%s: %s", p->name, ty);
    }
    return 0;
}

static void emit_nano_body(const NlNsi *nsi, const NlNsiMethod *m, FILE *out) {
    size_t j;
    const NlNsiParam *intp = NULL;
    const NlNsiParam *strp = NULL;
    const NlNsiParam *boolp = NULL;
    for (j = 0; j < m->param_count; j++) {
        const NlNsiParam *p = &m->params[j];
        const char *ty;
        if (!is_in_param(p)) continue;
        ty = nano_type(nsi, p->type_id);
        if (!intp && ty && strcmp(ty, "int") == 0) intp = p;
        if (!strp && ty && strcmp(ty, "string") == 0) strp = p;
        if (!boolp && ty && strcmp(ty, "bool") == 0) boolp = p;
    }
    if (intp && strp)
        fprintf(out, " return (cond ((== %s \"\") %s) (else %s))\n",
                strp->name, intp->name, intp->name);
    else if (intp)
        fprintf(out, " return %s\n", intp->name);
    else if (strp)
        fprintf(out, " return (cond ((== %s \"\") 0) (else 0))\n", strp->name);
    else if (boolp)
        fprintf(out, " return (cond (%s 1) (else 0))\n", boolp->name);
    else
        fprintf(out, " return 0\n");
}

int nl_nsi_gen_nanolang(const NlNsi *nsi, FILE *out) {
    size_t i;
    size_t j;
    char fn[256];
    if (!nsi || !out) return -1;
    fprintf(out, "# generated from %s — I do not infer a C ABI\n\n",
            nl_nsi_interface_id(nsi));
    for (i = 0; i < nsi->method_count; i++) {
        const NlNsiMethod *m = &nsi->methods[i];
        ident_copy(fn, sizeof(fn), m->id, '_');
        fprintf(out, "fn %s(", fn);
        if (emit_in_params_nano(nsi, m, out) != 0) return -1;
        fprintf(out, ") -> int {\n");
        emit_nano_body(nsi, m, out);
        fprintf(out, "}\n\n");
        fprintf(out, "shadow %s {\n assert (== (%s", fn, fn);
        for (j = 0; j < m->param_count; j++) {
            const NlNsiParam *p = &m->params[j];
            const char *ty;
            if (!is_in_param(p)) continue;
            ty = nano_type(nsi, p->type_id);
            fprintf(out, " %s", shadow_lit(ty));
        }
        fprintf(out, ") 0)\n}\n\n");
    }
    fprintf(out, "fn main() -> int {\n return 0\n}\n\n");
    fprintf(out, "shadow main {\n assert (== (main) 0)\n}\n");
    return 0;
}

int nl_nsi_gen_forth(const NlNsi *nsi, FILE *out) {
    size_t i;
    size_t j;
    char fn[256];
    if (!nsi || !out) return -1;
    fprintf(out, "\\ generated from %s — I do not infer a C ABI\n\n",
            nl_nsi_interface_id(nsi));
    for (i = 0; i < nsi->method_count; i++) {
        const NlNsiMethod *m = &nsi->methods[i];
        ident_copy(fn, sizeof(fn), m->id, '-');
        fprintf(out, ": %s ( ", fn);
        for (j = 0; j < m->param_count; j++) {
            const NlNsiParam *p = &m->params[j];
            const char *ty;
            if (!is_in_param(p)) continue;
            ty = nano_type(nsi, p->type_id);
            if (!ty) return -1;
            if (strcmp(ty, "string") == 0) fprintf(out, "c-addr u ");
            else fprintf(out, "n ");
        }
        fprintf(out, "-- ior ) ");
        for (j = 0; j < m->param_count; j++) {
            const NlNsiParam *p = &m->params[j];
            const char *ty;
            if (!is_in_param(p)) continue;
            ty = nano_type(nsi, p->type_id);
            if (strcmp(ty, "string") == 0) fprintf(out, "2drop ");
            else fprintf(out, "drop ");
        }
        fprintf(out, "0 ;\n");
    }
    return 0;
}

int nl_nsi_gen_python(const NlNsi *nsi, FILE *out) {
    size_t i;
    size_t j;
    char fn[256];
    if (!nsi || !out) return -1;
    fprintf(out, "# generated from %s — same contract as NanoLang/Forth/C\n",
            nl_nsi_interface_id(nsi));
    for (i = 0; i < nsi->method_count; i++) {
        const NlNsiMethod *m = &nsi->methods[i];
        int first = 1;
        ident_copy(fn, sizeof(fn), m->id, '_');
        fprintf(out, "def %s(", fn);
        for (j = 0; j < m->param_count; j++) {
            if (!is_in_param(&m->params[j])) continue;
            if (!first) fprintf(out, ", ");
            first = 0;
            fprintf(out, "%s", m->params[j].name);
        }
        fprintf(out, "):\n    # method %s\n    return 0\n\n", m->id);
    }
    return 0;
}

int nl_nsi_gen_rust(const NlNsi *nsi, FILE *out) {
    size_t i;
    size_t j;
    char fn[256];
    if (!nsi || !out) return -1;
    fprintf(out, "// generated from %s — same contract as NanoLang/Forth/C\n",
            nl_nsi_interface_id(nsi));
    for (i = 0; i < nsi->method_count; i++) {
        const NlNsiMethod *m = &nsi->methods[i];
        int first = 1;
        ident_copy(fn, sizeof(fn), m->id, '_');
        fprintf(out, "fn %s(", fn);
        for (j = 0; j < m->param_count; j++) {
            const NlNsiParam *p = &m->params[j];
            const char *ty;
            if (!is_in_param(p)) continue;
            ty = nano_type(nsi, p->type_id);
            if (!ty) return -1;
            if (!first) fprintf(out, ", ");
            first = 0;
            if (strcmp(ty, "string") == 0)
                fprintf(out, "%s: &str", p->name);
            else
                fprintf(out, "%s: i64", p->name);
        }
        fprintf(out, ") -> i32 { /* %s */ 0 }\n", m->id);
    }
    return 0;
}

int nl_nsi_gen_cxx(const NlNsi *nsi, FILE *out) {
    size_t i;
    size_t j;
    char fn[256];
    if (!nsi || !out) return -1;
    fprintf(out, "// generated from %s — same contract as NanoLang/Forth/C\n",
            nl_nsi_interface_id(nsi));
    for (i = 0; i < nsi->method_count; i++) {
        const NlNsiMethod *m = &nsi->methods[i];
        int first = 1;
        ident_copy(fn, sizeof(fn), m->id, '_');
        fprintf(out, "int %s(", fn);
        for (j = 0; j < m->param_count; j++) {
            const NlNsiParam *p = &m->params[j];
            const char *ty;
            if (!is_in_param(p)) continue;
            ty = nano_type(nsi, p->type_id);
            if (!ty) return -1;
            if (!first) fprintf(out, ", ");
            first = 0;
            if (strcmp(ty, "string") == 0)
                fprintf(out, "const char *%s", p->name);
            else
                fprintf(out, "int %s", p->name);
        }
        fprintf(out, ") { /* %s */ return 0; }\n", m->id);
    }
    return 0;
}

int nl_nsi_gen_dispatch(const NlNsi *nsi, FILE *out) {
    size_t i;
    if (!nsi || !out) return -1;
    fprintf(out, "/* generated dispatch for %s — method ids, not symbol names */\n",
            nl_nsi_interface_id(nsi));
    fprintf(out, "static const char *nsi_method_ids[] = {\n");
    for (i = 0; i < nsi->method_count; i++) {
        fprintf(out, "    \"%s\",\n", nsi->methods[i].id);
    }
    fprintf(out, "    0\n};\n");
    return 0;
}

int nl_nsi_gen_mock(const NlNsi *nsi, FILE *out) {
    return nl_nsi_gen_nanolang(nsi, out);
}

int nl_nsi_gen_docs(const NlNsi *nsi, FILE *out) {
    size_t i;
    size_t j;
    if (!nsi || !out) return -1;
    fprintf(out, "# %s\n\nI bind this interface by id, not by symbol name.\n\n",
            nl_nsi_interface_id(nsi));
    for (i = 0; i < nsi->method_count; i++) {
        fprintf(out, "- `%s`", nsi->methods[i].id);
        for (j = 0; j < nsi->methods[i].param_count; j++) {
            const NlNsiParam *p = &nsi->methods[i].params[j];
            fprintf(out, " `%s:%s`", p->name, p->type_id ? p->type_id : "?");
        }
        fprintf(out, "\n");
    }
    return 0;
}

int nl_nsi_gen_serialize(const NlNsi *nsi, FILE *out) {
    const char *iface;
    const char *method;
    if (!nsi || !out || nsi->method_count == 0) return -1;
    iface = nl_nsi_interface_id(nsi);
    method = nsi->methods[0].id;
    fprintf(out,
            "{\"nsi_version\":0,\"frame\":\"request\",\"interface\":\"%s\","
            "\"method\":\"%s\",\"call_id\":\"1\",\"payload\":{}}\n",
            iface, method);
    fprintf(out,
            "{\"nsi_version\":0,\"frame\":\"response\",\"call_id\":\"1\","
            "\"payload\":{\"ok\":true}}\n");
    return 0;
}

int nl_nsi_gen_validate(const NlNsi *nsi, FILE *out) {
    size_t i;
    if (!nsi || !out) return -1;
    fprintf(out, "/* validate %s by method id */\n", nl_nsi_interface_id(nsi));
    fprintf(out, "static int nsi_method_known(const char *id) {\n");
    for (i = 0; i < nsi->method_count; i++) {
        fprintf(out, "    if (strcmp(id, \"%s\") == 0) return 1;\n", nsi->methods[i].id);
    }
    fprintf(out, "    return 0;\n}\n");
    return 0;
}

int nl_nsi_gen_compat_tests(const NlNsi *nsi, FILE *out) {
    if (!nsi || !out) return -1;
    fprintf(out,
            "/* compatibility tests for %s: adding a method is OK; "
            "removing one is breaking */\n",
            nl_nsi_interface_id(nsi));
    fprintf(out, "/* use nl_nsi_compat(older, newer) */\n");
    return 0;
}

int nl_nsi_gen_language_index(const NlNsi *nsi, FILE *out) {
    size_t i;
    if (!nsi || !out || nsi->method_count == 0) return -1;
    fprintf(out, "C %s\n", nsi->methods[0].id);
    fprintf(out, "C++ %s\n", nsi->methods[0].id);
    fprintf(out, "Python %s\n", nsi->methods[0].id);
    fprintf(out, "Rust %s\n", nsi->methods[0].id);
    fprintf(out, "NanoLang %s\n", nsi->methods[0].id);
    fprintf(out, "NanoVM %s\n", nsi->methods[0].id);
    fprintf(out, "remote %s\n", nsi->methods[0].id);
    for (i = 1; i < nsi->method_count; i++) {
        fprintf(out, "C %s\n", nsi->methods[i].id);
    }
    return 0;
}

int nl_nsi_gen_nanoisa_imports(const NlNsi *nsi, FILE *out) {
    size_t i;
    size_t j;
    size_t nparams;
    if (!nsi || !out) return -1;
    fprintf(out, "# NanoISA import descriptors from %s\n", nl_nsi_interface_id(nsi));
    fprintf(out, "# module_id method_id param_count return=int (status)\n");
    for (i = 0; i < nsi->method_count; i++) {
        nparams = 0;
        for (j = 0; j < nsi->methods[i].param_count; j++) {
            if (is_in_param(&nsi->methods[i].params[j])) nparams++;
        }
        fprintf(out, "IMPORT %s %s params=%zu return=int\n",
                nl_nsi_interface_id(nsi), nsi->methods[i].id, nparams);
    }
    for (i = 0; i < nsi->capability_count; i++) {
        fprintf(out, "TRAP cap %s\n", nsi->capabilities[i].id);
    }
    return 0;
}
