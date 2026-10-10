#include "service_namespace.h"
#include "nanoisa/file_source_snapshot.h"
#include <stdlib.h>
#include <string.h>

#define NAME_LIMIT (NL_FILE_SOURCE_REQUESTS * NL_SERVICE_SOURCE_BINDINGS + NL_FILE_SOURCE_ALIASES + NL_FILE_SOURCE_ORDINARY)
#define MODULE_LIMIT 5000u
#define NONE NL_FILE_SOURCE_NO_INDEX
typedef struct { char *path; ASTNode *program; } SourceModule;
struct NlServiceNamespace {
    SourceModule *modules;
    size_t module_count, count, text_bytes, ordinary_count, alias_count;
    NlServiceName names[NAME_LIMIT];
    NlFileSourcePlan *plan;
};

static NlFileSourceText span(const char *s) {
    return (NlFileSourceText){s, s ? strlen(s) : 0};
}
static char *copy_text(NlServiceNamespace *space, const char *text) {
    if (!text) return NULL;
    size_t size = strlen(text);
    char *copy = malloc(size + 1);
    if (!copy) return NULL;
    memcpy(copy, text, size + 1);
    space->text_bytes += size + 1;
    return copy;
}
static uint32_t module_index(const NlServiceNamespace *space, const char *path) {
    for (size_t i = 0; i < space->module_count; ++i)
        if (!strcmp(space->modules[i].path, path)) return (uint32_t)i;
    return NONE;
}
static const NlServiceName *local_name(const NlServiceNamespace *space, uint32_t owner,
                                        const char *name, size_t length) {
    for (size_t i = 0; i < space->count; ++i) {
        const NlServiceName *row = &space->names[i];
        if (row->module == owner && strlen(row->name) == length && !memcmp(row->name, name, length)) return row;
    }
    return NULL;
}
static NlFileSourceStatus add_name(NlServiceNamespace *space, uint32_t module, const char *name,
                                  uint32_t kind, uint32_t ordinal, uint32_t request,
                                  uint32_t target, uint32_t target_module, bool exported,
                                  bool immutable, ASTNode *declaration) {
    if (!name || !*name || strlen(name) > 128) return NL_FILE_SOURCE_INVALID;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p)
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || *p == '_' ||
              (p != (const unsigned char *)name && *p >= '0' && *p <= '9'))) return NL_FILE_SOURCE_INVALID;
    const NlServiceName *prior = local_name(space, module, name, strlen(name));
    if (prior) {
        /* I coalesce repeated imports only when their original identity agrees. */
        bool repeated_module = kind == NL_SERVICE_MODULE && prior->kind == kind && prior->target_module == target_module;
        if (!repeated_module && (!target || prior->target != target || prior->kind != kind || prior->target_module != target_module))
            return NL_FILE_SOURCE_INVALID;
        if (exported) space->names[prior->id - 1].exported = true;
        return NL_FILE_SOURCE_OK;
    }
    bool alias = target || kind == NL_SERVICE_MODULE;
    if (space->count == NAME_LIMIT || (alias && space->alias_count == NL_FILE_SOURCE_ALIASES))
        return NL_FILE_SOURCE_LIMIT;
    bool ordinary = kind != NL_SERVICE_TYPE && kind != NL_SERVICE_METHOD;
    if (ordinary && space->ordinary_count == NL_FILE_SOURCE_ORDINARY) return NL_FILE_SOURCE_LIMIT;
    if (strlen(name) + 1 > NL_FILE_SOURCE_TEXT_BUDGET - space->text_bytes) return NL_FILE_SOURCE_LIMIT;
    char *owned = copy_text(space, name);
    if (!owned) return NL_FILE_SOURCE_MEMORY;
    uint32_t id = (uint32_t)space->count + 1;
    space->names[space->count++] = (NlServiceName){owned, id, target ? target : id,
        module, target_module, kind, ordinal, request, exported, immutable, declaration};
    if (ordinary) ++space->ordinary_count;
    if (alias) ++space->alias_count;
    return NL_FILE_SOURCE_OK;
}
static NlFileSourceStatus declarations(NlServiceNamespace *space, Environment *env, uint32_t module) {
    ASTNode *program = space->modules[module].program;
    for (int i = 0; i < program->as.program.count; ++i) {
        ASTNode *node = program->as.program.items[i];
        const char *name = NULL;
        uint32_t kind = 0;
        bool exported = true, immutable = false;
        if (node->type == AST_ASYNC_FN) node = node->as.async_fn.function;
        switch (node->type) {
            case AST_FUNCTION:
                if (!node->as.function.is_anonymous) { name = node->as.function.name; kind = NL_SERVICE_FUNCTION; }
                break;
            case AST_STRUCT_DEF: name = node->as.struct_def.original_name ? node->as.struct_def.original_name : node->as.struct_def.name; kind = NL_SERVICE_RECORD; break;
            case AST_UNION_DEF: name = node->as.union_def.name; kind = NL_SERVICE_UNION; break;
            case AST_ENUM_DEF: name = node->as.enum_def.name; kind = NL_SERVICE_ENUM; break;
            case AST_OPAQUE_TYPE: name = node->as.opaque_type.name; kind = NL_SERVICE_OPAQUE; break;
            case AST_LET: name = node->as.let.name; kind = NL_SERVICE_GLOBAL;
                exported = node->as.let.is_pub; immutable = !node->as.let.is_mut; break;
            case AST_SERVICE_DECL: {
                int64_t origin = node->as.service_decl.origin_index;
                if (origin < 0 || origin >= env->service_origin_count || !env->service_snapshot_bound[origin] ||
                    strcmp(env->service_origins[origin], space->modules[module].path)) return NL_FILE_SOURCE_INVALID;
                int64_t identity=nl_service_source_catalog_id(node->as.service_decl.interface_id);
                if(!identity || identity!=nl_service_source_snapshot_catalog(env->service_inputs,
                    env->service_snapshot_indices[origin]))return NL_FILE_SOURCE_UNRESOLVED;
                uint32_t types=(uint32_t)nl_service_source_catalog_count(identity,1);
                uint32_t methods=(uint32_t)nl_service_source_catalog_count(identity,2);
                for (uint32_t binding = 0; binding < types+methods; ++binding) {
                    uint32_t ordinal = binding < types ? binding : binding - types;
                    const char *symbol = nl_service_source_catalog_string(identity,binding < types ? 1 : 2, ordinal, 1, 0);
                    NlFileSourceStatus status = add_name(space, module, symbol,
                        binding < types ? NL_SERVICE_TYPE : NL_SERVICE_METHOD, ordinal, (uint32_t)origin,
                        0, module, true, false, node);
                    if (status) return status;
                }
                break;
            }
            default: break;
        }
        if (name) {
            NlFileSourceStatus status = add_name(space, module, name, kind, NONE, NONE, 0, module,
                                                exported, immutable, node);
            if (status) return status;
        }
    }
    return NL_FILE_SOURCE_OK;
}
static NlFileSourceStatus import_name(NlServiceNamespace *space, uint32_t owner,
                                      const char *name, const NlServiceName *target, bool exported) {
    return add_name(space, owner, name, target->kind, target->ordinal, target->request,
                    target->target, target->target_module, exported, target->immutable, target->declaration);
}
static NlFileSourceStatus imports(NlServiceNamespace *space, uint32_t owner) {
    ASTNode *program = space->modules[owner].program;
    for (int i = 0; i < program->as.program.count; ++i) {
        ASTNode *node = program->as.program.items[i];
        if (node->type != AST_IMPORT) continue;
        char *resolved = (char *)resolve_module_path(node->as.import_stmt.module_path, space->modules[owner].path);
        char *canonical = resolved ? realpath(resolved, NULL) : NULL;
        free(resolved);
        if (!canonical) return NL_FILE_SOURCE_UNRESOLVED;
        uint32_t target_module = module_index(space, canonical);
        free(canonical);
        /* I require the loader's dependency-first graph, including re-exports. */
        if (target_module == NONE || target_module >= owner) return NL_FILE_SOURCE_UNRESOLVED;
        const char *prefix = node->as.import_stmt.module_alias;
        if (prefix && *prefix) {
            NlFileSourceStatus status = add_name(space, owner, prefix, NL_SERVICE_MODULE, NONE, NONE,
                                                0, target_module, node->as.import_stmt.is_pub_use, false, node);
            if (status) return status;
            continue;
        }
        if (node->as.import_stmt.is_selective && !node->as.import_stmt.is_wildcard) {
            for (int selected = 0; selected < node->as.import_stmt.import_symbol_count; ++selected) {
                const char *name = node->as.import_stmt.import_symbols[selected];
                const NlServiceName *target = name ? local_name(space, target_module, name, strlen(name)) : NULL;
                if (!target || !target->exported) return NL_FILE_SOURCE_UNRESOLVED;
                const char *alias = node->as.import_stmt.import_aliases ? node->as.import_stmt.import_aliases[selected] : NULL;
                NlFileSourceStatus status = import_name(space, owner, alias && *alias ? alias : name,
                                                       target, node->as.import_stmt.is_pub_use);
                if (status) return status;
            }
        } else {
            size_t count = space->count;
            for (size_t n = 0; n < count; ++n) {
                const NlServiceName *target = &space->names[n];
                bool legacy_constant = !node->as.import_stmt.is_selective && target->kind == NL_SERVICE_GLOBAL && target->immutable;
                if (target->module != target_module || (!target->exported && !legacy_constant)) continue;
                NlFileSourceStatus status = import_name(space, owner, target->name, target, node->as.import_stmt.is_pub_use);
                if (status) return status;
            }
        }
    }
    return NL_FILE_SOURCE_OK;
}
static NlFileSourceStatus describe(NlServiceNamespace *space, Environment *env) {
    char catalog_view[2][32768];size_t catalog_size[2]={0};
    for(int64_t c=1;c<=2;c++)
        if(!nl_service_source_catalog_view(c,catalog_view[c-1],sizeof catalog_view[0],&catalog_size[c-1]))return NL_FILE_SOURCE_UNRESOLVED;
    NlFileSourceRequest requests[NL_FILE_SOURCE_REQUESTS] = {0};
    NlFileSourceBinding bindings[NL_FILE_SOURCE_REQUESTS][NL_SERVICE_SOURCE_BINDINGS] = {0};
    NlFileSourceAlias aliases[NL_FILE_SOURCE_ALIASES];
    NlFileSourceOrdinary ordinary[NL_FILE_SOURCE_ORDINARY];
    size_t counts[NL_FILE_SOURCE_REQUESTS] = {0}, nr = (size_t)env->service_origin_count, na = 0, no = 0;
    if (!nr || nr > NL_FILE_SOURCE_REQUESTS) return NL_FILE_SOURCE_INVALID;
    for (size_t i = 0; i < space->count; ++i) {
        const NlServiceName *name = &space->names[i];
        if (name->kind != NL_SERVICE_TYPE && name->kind != NL_SERVICE_METHOD) {
            if (no == NL_FILE_SOURCE_ORDINARY) return NL_FILE_SOURCE_LIMIT;
            ordinary[no++] = (NlFileSourceOrdinary){span(space->modules[name->module].path), span(name->name), name->id};
        } else if (name->id != name->target) {
            if (na == NL_FILE_SOURCE_ALIASES) return NL_FILE_SOURCE_LIMIT;
            aliases[na++] = (NlFileSourceAlias){span(space->modules[name->module].path), span(name->name), name->id, name->target};
        } else {
            uint32_t r = name->request;
            if (r >= nr || counts[r] == NL_SERVICE_SOURCE_BINDINGS) return NL_FILE_SOURCE_INVALID;
            bindings[r][counts[r]++] = (NlFileSourceBinding){name->id, name->kind == NL_SERVICE_METHOD, name->ordinal, span(name->name)};
            ASTNode *node = name->declaration;
            size_t length = 0;
            const unsigned char *catalog = nl_file_source_snapshot_bytes(env->service_inputs,
                env->service_snapshot_indices[r], 2, &length);
            if (!catalog || !length) return NL_FILE_SOURCE_UNRESOLVED;
            int64_t identity=nl_service_source_snapshot_catalog(env->service_inputs,env->service_snapshot_indices[r]);
            if(identity<1 || identity>2 || identity!=nl_service_source_catalog_id(node->as.service_decl.interface_id))return NL_FILE_SOURCE_UNRESOLVED;
            size_t expected=(size_t)(nl_service_source_catalog_count(identity,1)+nl_service_source_catalog_count(identity,2));
            requests[r] = (NlFileSourceRequest){span(space->modules[name->module].path), span(node->as.service_decl.interface_id),
                {catalog_view[identity-1], catalog_size[identity-1] - 1}, (uint32_t)node->as.service_decl.catalog_version,
                (uint32_t)node->line, (uint32_t)node->column, bindings[r], expected};
        }
    }
    for (size_t r = 0; r < nr; ++r) if (counts[r] != requests[r].binding_count || !counts[r]) return NL_FILE_SOURCE_INVALID;
    return nl_service_source_plan_build(requests, nr, aliases, na, ordinary, no, &space->plan);
}

NlFileSourceStatus nl_service_namespace_build(ASTNode *root, Environment *env, ModuleList *modules,
                                              const char *path, NlServiceNamespace **out) {
    if (!root || root->type != AST_PROGRAM || !env || !path || !out ||
        (modules && modules->count < 0)) return NL_FILE_SOURCE_INVALID;
    size_t count = modules ? (size_t)modules->count : 0;
    if (count >= MODULE_LIMIT) return NL_FILE_SOURCE_LIMIT;
    NlServiceNamespace *space = calloc(1, sizeof *space);
    if (!space) return NL_FILE_SOURCE_MEMORY;
    space->modules = calloc(count + 1, sizeof *space->modules);
    if (!space->modules) { nl_service_namespace_free(space); return NL_FILE_SOURCE_MEMORY; }
    NlFileSourceStatus status = NL_FILE_SOURCE_OK;
    for (size_t i = 0; i <= count; ++i) {
        const char *input = i == count ? path : modules->module_paths[i];
        char *canonical = input ? realpath(input, NULL) : NULL;
        ASTNode *program = i == count ? root : get_cached_module_ast(input);
        if (!canonical || !program || program->type != AST_PROGRAM || module_index(space, canonical) != NONE) {
            free(canonical); status = NL_FILE_SOURCE_INVALID; goto done;
        }
        if (!*canonical || strlen(canonical) > 4096) {
            free(canonical); status = NL_FILE_SOURCE_INVALID; goto done;
        }
        if (strlen(canonical) + 1 > NL_FILE_SOURCE_TEXT_BUDGET - space->text_bytes) {
            free(canonical); status = NL_FILE_SOURCE_LIMIT; goto done;
        }
        char *owned = copy_text(space, canonical);
        free(canonical);
        if (!owned) { status = NL_FILE_SOURCE_MEMORY; goto done; }
        space->modules[space->module_count++] = (SourceModule){owned, program};
    }
    for (size_t i = 0; i < space->module_count; ++i) {
        status = declarations(space, env, (uint32_t)i);
        if (status) goto done;
    }
    for (size_t i = 0; i < space->module_count; ++i) {
        status = imports(space, (uint32_t)i);
        if (status) goto done;
    }
    status = describe(space, env);
done:
    if (status) nl_service_namespace_free(space);
    else *out = space;
    return status;
}
void nl_service_namespace_free(NlServiceNamespace *space) {
    if (!space) return;
    for (size_t i = 0; i < space->count; ++i) free((void *)space->names[i].name);
    for (size_t i = 0; i < space->module_count; ++i) free(space->modules[i].path);
    free(space->modules);
    nl_file_source_plan_free(space->plan);
    free(space);
}
size_t nl_service_namespace_count(const NlServiceNamespace *space) { return space ? space->count : 0; }
const NlServiceName *nl_service_namespace_name(const NlServiceNamespace *space, size_t index) {
    return space && index < space->count ? &space->names[index] : NULL;
}
const char *nl_service_namespace_module(const NlServiceNamespace *space, uint32_t module) {
    return space && module < space->module_count ? space->modules[module].path : NULL;
}
const NlFileSourcePlan *nl_service_namespace_plan(const NlServiceNamespace *space) { return space ? space->plan : NULL; }
const NlServiceName *nl_service_namespace_lookup(const NlServiceNamespace *space, const char *module, const char *name) {
    if (!space || !module || !name || !*name) return NULL;
    uint32_t owner = module_index(space, module);
    bool imported = false;
    for (size_t depth = 0; owner != NONE && depth <= space->module_count; ++depth) {
        const char *dot = strchr(name, '.');
        size_t length = dot ? (size_t)(dot - name) : strlen(name);
        const NlServiceName *row = local_name(space, owner, name, length);
        if (!row || (imported && !row->exported)) return NULL;
        if (!dot) return row;
        if (row->kind != NL_SERVICE_MODULE || !dot[1]) return NULL;
        owner = row->target_module; name = dot + 1; imported = true;
    }
    return NULL;
}


int64_t nl_service_namespace_catalog(const NlServiceNamespace *space,uint32_t module) {
    if(!space || module>=space->module_count)return NL_SOURCE_CATALOG_NONE;
    for(size_t i=0;i<nl_file_source_plan_count(space->plan);i++) {
        NlFileSourceRow row;
        if(nl_file_source_plan_row(space->plan,i,&row) && row.id==row.target &&
           row.module.size==strlen(space->modules[module].path) &&
           !memcmp(row.module.data,space->modules[module].path,row.module.size)) return row.catalog;
    }
    return NL_SOURCE_CATALOG_NONE;
}
static bool catalog_type(const NlServiceNamespace *space, uint32_t module,
                         uint32_t ordinal, TypeInfo *out) {
    int64_t identity=nl_service_namespace_catalog(space,module);
    if (!identity || !out || ordinal >= (uint32_t)nl_service_source_catalog_count(identity,1)) return false;
    const char *name = nl_service_source_catalog_string(identity,1, ordinal, 1, 0);
    const NlServiceName *row = local_name(space, module, name, strlen(name));
    if (!row || row->kind != NL_SERVICE_TYPE || row->id != row->target ||
        row->ordinal != ordinal || row->target_module != module) return false;
    TypeInfo result = {0};
    result.base_type = ordinal == 0 ? TYPE_OPAQUE : (ordinal < 3 || (identity==2 && ordinal==8)) ? TYPE_STRUCT : TYPE_UNION;
    result.service_declaration = row->target;
    result.service_module = module;
    result.service_ordinal = ordinal;
    result.service_category = ordinal == 0 ? 1 : ordinal == 3 ? 2 : (ordinal < 3 || (identity==2 && ordinal==8)) ? 3 : 4;
    *out = result;
    return true;
}
bool nl_service_type(const NlServiceNamespace *space, const char *module,
                     const char *name, TypeInfo *out) {
    if (!out) return false;
    const NlServiceName *row = nl_service_namespace_lookup(space, module, name);
    if (!row || row->kind != NL_SERVICE_TYPE) return false;
    return catalog_type(space, row->target_module, row->ordinal, out);
}
static bool catalog_value_type(const NlServiceNamespace *space, uint32_t module,
                               const char *id, TypeInfo *out) {
    if (!id || !out) return false;
    int64_t identity=nl_service_namespace_catalog(space,module);
    TypeInfo scalar = {0};
    if (!*id) scalar.base_type = TYPE_VOID;
    else if (!strcmp(id, "nsi:core/int")) scalar.base_type = TYPE_INT;
    else if (!strcmp(id, "nsi:core/bool")) scalar.base_type = TYPE_BOOL;
    else {
        for (uint32_t ordinal = 0; ordinal < (uint32_t)(identity?nl_service_source_catalog_count(identity,1):0); ++ordinal)
            if (!strcmp(id, nl_service_source_catalog_string(identity,1, ordinal, 0, 0)))
                return catalog_type(space, module, ordinal, out);
        return false;
    }
    *out = scalar;
    return true;
}
bool nl_service_member_type(const NlServiceNamespace *space, const TypeInfo *owner,
                            const char *member, TypeInfo *out) {
    if (!owner || !member || !out || !owner->service_declaration) return false;
    TypeInfo expected;
    if (!catalog_type(space, owner->service_module, owner->service_ordinal, &expected) ||
        !type_infos_equal(owner, &expected)) return false;
    int64_t identity=nl_service_namespace_catalog(space,owner->service_module);
    int64_t count = nl_service_source_catalog_number(identity,1, owner->service_ordinal, 1, 0);
    for (int64_t i = 0; i < count; ++i)
        if (!strcmp(member, nl_service_source_catalog_string(identity,1, owner->service_ordinal, 3, i)))
            return catalog_value_type(space, owner->service_module,
                nl_service_source_catalog_string(identity,1, owner->service_ordinal, 4, i), out);
    return false;
}
bool nl_service_method_type(const NlServiceNamespace *space, const char *module,
                            const char *name, NlServiceSignature *out) {
    if (!out) return false;
    const NlServiceName *row = nl_service_namespace_lookup(space, module, name);
    if (!row || row->kind != NL_SERVICE_METHOD || row->ordinal >= 5) return false;
    int64_t identity=nl_service_namespace_catalog(space,row->target_module);
    if(!identity)return false;
    NlServiceSignature result = {0};
    result.declaration = row->target; result.module = row->target_module;
    result.ordinal = row->ordinal;
    result.input_mode = (uint32_t)nl_service_source_catalog_number(identity,2, row->ordinal, 3, 0);
    int64_t count = nl_service_source_catalog_number(identity,2, row->ordinal, 4, 0);
    /* My catalog places the result after its zero, one or two inputs. */
    if (count < 1 || count > 3) return false;
    result.parameter_count = (uint32_t)count - 1;
    for (int64_t i = 0; i < count; ++i) {
        TypeInfo *target = i == count - 1 ? &result.result : &result.parameters[i];
        if (!catalog_value_type(space, row->target_module,
                nl_service_source_catalog_string(identity,2, row->ordinal, 6, i), target)) return false;
    }
    *out = result;
    return true;
}

ASTNode *nl_service_namespace_program(const NlServiceNamespace *space, uint32_t module) {
    return space && module < space->module_count ? space->modules[module].program : NULL;
}
