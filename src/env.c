#include "nanolang.h"
#include "builtins_registry.h"
#include "runtime/gc.h"
#include <string.h>

typedef struct {
    uint64_t hash;
    int previous; /* I encode slot + 1; -1 marks a slot without a name. */
} EnvSymbolLink;

struct EnvSymbolIndex {
    int *heads;
    EnvSymbolLink *links;
    size_t bucket_count, capacity;
    int count;
};

void env_symbol_index_invalidate(Environment *env) {
    if (!env || !env->symbol_index) return;
    free(env->symbol_index->heads);
    free(env->symbol_index->links);
    free(env->symbol_index);
    env->symbol_index = NULL;
}

static uint64_t symbol_name_hash(const char *name) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        hash ^= *p;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

/* I store indices and hashes, not borrowed names or Symbol pointers. Scope
 * cleanup may free names before lowering symbol_count; popping only needs the
 * saved links. Normal insertion synchronizes before reusing a popped slot. */
static struct EnvSymbolIndex *symbol_index_sync(Environment *env) {
    struct EnvSymbolIndex *index = env->symbol_index;
    if (!index) {
        index = calloc(1, sizeof *index);
        if (!index) return NULL;
        env->symbol_index = index;
    }
    while (index->count > env->symbol_count) {
        EnvSymbolLink *link = &index->links[--index->count];
        if (link->previous >= 0)
            index->heads[link->hash & (index->bucket_count - 1)] = link->previous;
    }
    size_t needed = (size_t)env->symbol_count;
    if (needed > index->capacity) {
        size_t capacity = index->capacity ? index->capacity : 16;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2) goto unavailable;
            capacity *= 2;
        }
        if (capacity > SIZE_MAX / sizeof *index->links) goto unavailable;
        EnvSymbolLink *links = realloc(index->links, capacity * sizeof *links);
        if (!links) goto unavailable;
        index->links = links;
        index->capacity = capacity;
    }
    if (!index->bucket_count || needed > index->bucket_count / 2) {
        size_t buckets = index->bucket_count ? index->bucket_count : 32;
        while (needed > buckets / 2) {
            if (buckets > SIZE_MAX / 2) goto unavailable;
            buckets *= 2;
        }
        if (buckets > SIZE_MAX / sizeof *index->heads) goto unavailable;
        int *heads = calloc(buckets, sizeof *heads);
        if (!heads) goto unavailable;
        for (int i = 0; i < index->count; ++i) {
            EnvSymbolLink *link = &index->links[i];
            if (link->previous < 0) continue;
            size_t bucket = link->hash & (buckets - 1);
            link->previous = heads[bucket];
            heads[bucket] = i + 1;
        }
        free(index->heads);
        index->heads = heads;
        index->bucket_count = buckets;
    }
    while (index->count < env->symbol_count) {
        int slot = index->count++;
        EnvSymbolLink *link = &index->links[slot];
        const char *name = env->symbols[slot].name;
        link->previous = -1;
        if (!name) continue;
        link->hash = symbol_name_hash(name);
        size_t bucket = link->hash & (index->bucket_count - 1);
        link->previous = index->heads[bucket];
        index->heads[bucket] = slot + 1;
    }
    return index;

unavailable:
    /* I retain correct lookup when an optional index allocation fails. */
    env_symbol_index_invalidate(env);
    return NULL;
}

static Symbol *symbol_lookup(Environment *env, const char *name, bool same_file) {
    if (!env || !name) return NULL;
    struct EnvSymbolIndex *index = symbol_index_sync(env);
    uint64_t hash = symbol_name_hash(name);
    int next = index ? index->heads[hash & (index->bucket_count - 1)] : env->symbol_count;
    while (next) {
        int slot = next - 1;
        Symbol *sym = &env->symbols[slot];
        next = index ? index->links[slot].previous : slot;
        if (index && index->links[slot].hash != hash) continue;
        if (!sym->name || safe_strcmp(sym->name, name) != 0) continue;
        if (!same_file || sym->def_file == env->current_file ||
            (sym->def_file && env->current_file && strcmp(sym->def_file, env->current_file) == 0))
            return sym;
    }
    return NULL;
}

/* I retain checker allocations independently of mutable symbol/function slots.
 * Every registered block is unique and shallowly freed; borrowed subgraphs are
 * never traversed. This deliberately does not change runtime value ownership. */
struct EnvCheckerAllocation {
    void *allocation;
    struct EnvCheckerAllocation *next;
};
void *env_own_checker_allocation(Environment *env, void *allocation) {
    if (!allocation) return NULL;
    struct EnvCheckerAllocation *entry = malloc(sizeof *entry);
    if (!entry) {
        fprintf(stderr, "I could not allocate checker ownership metadata\n");
        exit(1);
    }
    entry->allocation = allocation;
    entry->next = env->checker_allocations;
    env->checker_allocations = entry;
    return allocation;
}

/* Create environment */
Environment *create_environment(void) {
    /* calloc, not malloc: every field below is set explicitly, but zeroing
     * first means a field added to Environment without a matching line here
     * defaults to false/NULL/0 instead of holding allocator garbage. Four
     * fields had already drifted that way -- see the diagnostics block. */
    Environment *env = calloc(1, sizeof(Environment));
    env->symbols = malloc(sizeof(Symbol) * 8);
    env->symbol_count = 0;
    env->symbol_capacity = 8;
    env->functions = malloc(sizeof(Function) * 8);
    env->function_count = 0;
    env->function_capacity = 8;
    env->structs = malloc(sizeof(StructDef) * 8);
    env->struct_count = 0;
    env->struct_capacity = 8;
    env->enums = calloc(8, sizeof(EnumDef));  /* Use calloc to zero-initialize */
    env->enum_count = 0;
    env->enum_capacity = 8;
    env->unions = malloc(sizeof(UnionDef) * 8);
    env->union_count = 0;
    env->union_capacity = 8;
    env->opaque_types = malloc(sizeof(OpaqueTypeDef) * 8);
    env->opaque_type_count = 0;
    env->opaque_type_capacity = 8;
    env->effects = NULL;
    env->effect_count = 0;
    env->effect_capacity = 0;
    env->generic_instances = malloc(sizeof(GenericInstantiation) * 8);
    env->generic_instance_count = 0;
    env->generic_instance_capacity = 8;
    env->generic_func_instances = malloc(sizeof(GenericFuncInstance) * 8);
    env->generic_func_instance_count = 0;
    env->generic_func_instance_capacity = 8;
    env->namespaces = malloc(sizeof(ModuleNamespace) * 8);
    env->namespace_count = 0;
    env->namespace_capacity = 8;
    env->current_module = NULL;  /* Start in global scope */
    
    /* Initialize module tracking for introspection */
    env->modules = NULL;
    env->module_count = 0;
    env->module_capacity = 0;
    env->emit_module_metadata = true;
    env->emit_c_main = true;
    env->current_module_is_unsafe = false;
    
    /* Initialize Phase 3: Module safety warning flags */
    env->warn_unsafe_imports = false;
    env->warn_unsafe_calls = false;
    env->warn_ffi = false;
    env->forbid_unsafe = false;
    env->profile = false;
    env->profile_gprof = false;
    env->profile_output_path = NULL;
    env->trace = false;
    env->profile_runtime = false;
    env->profile_flamegraph_path = NULL;
    env->gpu_target = false;
    env->suppress_shadow_warnings = false;
    
    /* Initialize import tracker */
    env->import_tracker = malloc(sizeof(ImportTracker));
    env->import_tracker->imports = malloc(sizeof(SelectiveImport) * 8);
    env->import_tracker->import_count = 0;
    env->import_tracker->import_capacity = 8;
    
    env->builtins_registered = false;
    env->effect_registry = NULL;

    return env;
}

static void env_free_value(Value v) {
    if (v.type == VAL_STRING) {
        /* Release GC-managed string if applicable */
        if (gc_is_managed(v.as.string_val)) {
            gc_release(v.as.string_val);
        } else {
            free(v.as.string_val);
        }
        return;
    }
    /* Note: Opaque pointers stored as VAL_INT are NOT released here in interpreter mode
     * because interpreter function-local variables persist in global environment.
     * GC cycle collection will clean them up when the program ends.
     * Compiled code handles opaque lifetimes correctly via scope-based cleanup. */
    if (v.type == VAL_STRUCT) {
        StructValue *sv = v.as.struct_val;
        if (!sv) return;
        free(sv->struct_name);
        for (int j = 0; j < sv->field_count; j++) {
            free(sv->field_names[j]);
            if (sv->field_values[j].type == VAL_STRING) {
                /* Release GC-managed strings in struct fields */
                if (gc_is_managed(sv->field_values[j].as.string_val)) {
                    gc_release(sv->field_values[j].as.string_val);
                } else {
                    free(sv->field_values[j].as.string_val);
                }
            }
        }
        free(sv->field_names);
        free(sv->field_values);
        free(sv);
        return;
    }
    if (v.type == VAL_FUNCTION) {
        if (v.as.function_val.function_name) {
            free((char*)v.as.function_val.function_name);
        }
        if (v.as.function_val.signature) {
            free_function_signature(v.as.function_val.signature);
        }
        return;
    }
}

/* I release only the symbols discarded by an existing lexical-scope pop. */
void env_restore_symbol_count(Environment *env, int count) {
    if (!env || count < 0 || count > env->symbol_count) return;
    while (env->symbol_count > count) {
        Symbol *symbol = &env->symbols[--env->symbol_count];
        free(symbol->name);
        free(symbol->struct_type_name);
        if (symbol->type != TYPE_BORROW_SHARED && symbol->type != TYPE_BORROW_MUT)
            env_free_value(symbol->value);
        memset(symbol, 0, sizeof *symbol);
    }
}

/* Free environment */
void free_environment(Environment *env) {
    env_symbol_index_invalidate(env);
    env_function_index_invalidate(env);
    env_restore_symbol_count(env, 0);
    free(env->symbols);
    if (env->import_tracker) {
        free(env->import_tracker->imports);
        free(env->import_tracker);
    }

    for (int i = 0; i < env->function_count; i++) {
        /* Note: function names are not owned by environment - they point to AST */
        /* Freeing them causes double-free crashes */
    }
    free(env->functions);
    
    for (int i = 0; i < env->struct_count; i++) {
        free(env->structs[i].name);
        free(env->structs[i].original_name);
        for (int j = 0; j < env->structs[i].field_count; j++) {
            free(env->structs[i].field_names[j]);
            if (env->structs[i].field_type_names)
                free(env->structs[i].field_type_names[j]);
        }
        free(env->structs[i].field_names);
        free(env->structs[i].field_types);
        free(env->structs[i].field_type_names);
        free(env->structs[i].field_element_types);
        /* Complete field annotations remain borrowed from the AST. */
    }
    free(env->structs);
    
    for (int i = 0; i < env->enum_count; i++) {
        free(env->enums[i].name);
        for (int j = 0; j < env->enums[i].variant_count; j++) {
            free(env->enums[i].variant_names[j]);
        }
        free(env->enums[i].variant_names);
        free(env->enums[i].variant_values);
    }
    free(env->enums);
    
    /* Free unions */
    for (int i = 0; i < env->union_count; i++) {
        free(env->unions[i].name);
        for (int j = 0; j < env->unions[i].variant_count; j++) {
            free(env->unions[i].variant_names[j]);
            if (env->unions[i].variant_field_type_info && env->unions[i].variant_field_type_info[j]) {
                for (int k = 0; k < env->unions[i].variant_field_counts[j]; ++k)
                    free_payload_type_info(env->unions[i].variant_field_type_info[j][k]);
                free(env->unions[i].variant_field_type_info[j]);
            }
            if (env->unions[i].variant_field_names && env->unions[i].variant_field_names[j]) {
                for (int k = 0; k < env->unions[i].variant_field_counts[j]; k++) {
                    free(env->unions[i].variant_field_names[j][k]);
                }
                free(env->unions[i].variant_field_names[j]);
            }
            if (env->unions[i].variant_field_types && env->unions[i].variant_field_types[j]) {
                free(env->unions[i].variant_field_types[j]);
            }
            if (env->unions[i].variant_field_type_names && env->unions[i].variant_field_type_names[j]) {
                for (int k = 0; k < env->unions[i].variant_field_counts[j]; k++)
                    free(env->unions[i].variant_field_type_names[j][k]);
                free(env->unions[i].variant_field_type_names[j]);
            }
        }
        free(env->unions[i].variant_field_type_info);
        if (env->unions[i].variant_names) free(env->unions[i].variant_names);
        if (env->unions[i].variant_field_counts) free(env->unions[i].variant_field_counts);
        if (env->unions[i].variant_field_names) free(env->unions[i].variant_field_names);
        if (env->unions[i].variant_field_types) free(env->unions[i].variant_field_types);
        free(env->unions[i].variant_field_type_names);
        for (int j = 0; j < env->unions[i].generic_param_count; j++)
            free(env->unions[i].generic_params[j]);
        free(env->unions[i].generic_params);
        free(env->unions[i].module_name);
    }
    free(env->unions);
    
    /* I own effect declaration copies, but their parameter type annotations
     * still borrow the AST. Only copied parameter names are mine. */
    for (int i = 0; i < env->effect_count; ++i) {
        EffectDef *effect = &env->effects[i];
        free(effect->name);
        free(effect->module_name);
        for (int j = 0; j < effect->op_count; ++j) {
            EffectOp *op = &effect->ops[j];
            free(op->name);
            free(op->return_type_name);
            for (int k = 0; k < op->param_count; ++k) free(op->params[k].name);
            free(op->params);
        }
        free(effect->ops);
    }
    free(env->effects);

    /* Free generic instantiations */
    for (int i = 0; i < env->generic_instance_count; i++) {
        free(env->generic_instances[i].generic_name);
        free(env->generic_instances[i].type_args);
        free(env->generic_instances[i].concrete_name);
        free_payload_type_info(env->generic_instances[i].type_info);
        if (env->generic_instances[i].type_arg_names) {
            for (int j = 0; j < env->generic_instances[i].type_arg_count; j++) {
                if (env->generic_instances[i].type_arg_names[j]) {
                    free(env->generic_instances[i].type_arg_names[j]);
                }
            }
            free(env->generic_instances[i].type_arg_names);
        }
    }
    free(env->generic_instances);

    /* Free generic function instances */
    if (env->generic_func_instances) {
        for (int i = 0; i < env->generic_func_instance_count; i++) {
            GenericFuncInstance *inst = &env->generic_func_instances[i];
            free(inst->orig_name);
            free(inst->mono_name);
            for (int j = 0; j < inst->binding_count; j++) {
                free(inst->var_names[j]);
                if (inst->bound_type_names[j]) free(inst->bound_type_names[j]);
            }
            free(inst->var_names);
            free(inst->bound_types);
            free(inst->bound_type_names);
        }
        free(env->generic_func_instances);
    }

    /* Free opaque types */
    for (int i = 0; i < env->opaque_type_count; i++) {
        free(env->opaque_types[i].name);
        free(env->opaque_types[i].c_type_name);
    }
    free(env->opaque_types);

    /* Free namespaces */
    for (int i = 0; i < env->namespace_count; i++) {
        free(env->namespaces[i].alias);
        free(env->namespaces[i].owner_module);
        free(env->namespaces[i].module_name);
        for (int j = 0; j < env->namespaces[i].function_count; j++) {
            free(env->namespaces[i].function_names[j]);
        }
        free(env->namespaces[i].function_names);
        for (int j = 0; j < env->namespaces[i].struct_count; j++) {
            free(env->namespaces[i].struct_names[j]);
        }
        free(env->namespaces[i].struct_names);
        for (int j = 0; j < env->namespaces[i].enum_count; j++) {
            free(env->namespaces[i].enum_names[j]);
        }
        free(env->namespaces[i].enum_names);
        for (int j = 0; j < env->namespaces[i].union_count; j++) {
            free(env->namespaces[i].union_names[j]);
        }
        free(env->namespaces[i].union_names);
    }
    free(env->namespaces);

    /* Free module metadata */
    if (env->modules) {
        for (int i = 0; i < env->module_count; i++) {
            free(env->modules[i].name);
            free(env->modules[i].path);
            if (env->modules[i].exported_functions) {
                for (int j = 0; j < env->modules[i].function_count; j++) {
                    free(env->modules[i].exported_functions[j]);
                }
                free(env->modules[i].exported_functions);
            }
            if (env->modules[i].exported_structs) {
                for (int j = 0; j < env->modules[i].struct_count; j++) {
                    free(env->modules[i].exported_structs[j]);
                }
                free(env->modules[i].exported_structs);
            }
        }
        free(env->modules);
    }

    while (env->checker_allocations) {
        struct EnvCheckerAllocation *entry = env->checker_allocations;
        env->checker_allocations = entry->next;
        free(entry->allocation);
        free(entry);
    }
    free(env);
}

/* Define variable */
void env_define_var(Environment *env, const char *name, Type type, bool is_mut, Value value) {
    env_define_var_with_element_type(env, name, type, TYPE_UNKNOWN, is_mut, value);
}

void env_define_var_with_element_type(Environment *env, const char *name, Type type, Type element_type, bool is_mut, Value value) {
    env_define_var_with_type_info(env, name, type, element_type, NULL, is_mut, value);
}

/* The most recent symbol with this name defined in the file currently being
 * processed. Used where "the same variable, seen again" is the question --
 * which is only ever true within one file. Matching by name alone lets a
 * definition inherit metadata from an unrelated symbol in another module,
 * which is the same cross-file confusion that made source-position lookups
 * wrong. */
static Symbol *env_get_var_same_file(Environment *env, const char *name) {
    return symbol_lookup(env, name, true);
}

void env_define_var_with_type_info(Environment *env, const char *name, Type type, Type element_type, TypeInfo *type_info, bool is_mut, Value value) {
    /* Borrowed parameters retain their caller's identity and do not own its storage. */
    if (value.type == VAL_STRUCT && value.as.struct_val &&
        type != TYPE_BORROW_SHARED && type != TYPE_BORROW_MUT) {
        StructValue *sv = value.as.struct_val;
        value = create_struct(sv->struct_name, sv->field_names, sv->field_values, sv->field_count);
    }
    if (env->symbol_count >= env->symbol_capacity) {
        env->symbol_capacity *= 2;
        env->symbols = realloc(env->symbols, sizeof(Symbol) * env->symbol_capacity);
    }

    Symbol sym;
    sym.name = strdup(name);
    sym.type = type;
    sym.struct_type_name = NULL;  /* Initialize to NULL (set later for struct types) */
    sym.element_type = element_type;  /* Store element type for arrays */
    sym.type_info = type_info;  /* Store full type info for complex types (tuples, etc.) */
    sym.is_mut = is_mut;
    sym.value = value;
    sym.is_global = false;
    sym.is_used = false;  /* Initialize as unused */
    sym.is_resource = false;  /* Will be set by type checker if type is a resource struct */
    sym.resource_state = RESOURCE_UNUSED;  /* Initialize resource state */
    sym.from_c_header = false;  /* Not from C header (normal nanolang variable) */
    sym.def_line = 0;     /* Will be set by type checker if needed */
    sym.def_column = 0;
    sym.flow_start_line = 0;
    sym.flow_start_column = 0;
    sym.scope_end_line = 0;
    sym.scope_end_column = 0;
    sym.def_file = env->current_file;   /* NULL when no file is in scope */

    /* WORKAROUND: Check if symbol already exists and preserve/update metadata */
    /* This handles a bug where symbols are added multiple times during type-checking.
     * When a symbol is re-added, preserve or update struct_type_name to maintain type information. */
    Symbol *existing = env_get_var_same_file(env, name);
    if (existing) {
        /* If existing has struct_type_name but new one doesn't, preserve it */
        if (existing->struct_type_name && !sym.struct_type_name) {
            sym.struct_type_name = strdup(existing->struct_type_name);
        }
        /* If new one has struct_type_name but existing doesn't, update the existing symbol instead */
        else if (!existing->struct_type_name && sym.struct_type_name) {
            /* Update the existing symbol with the new metadata */
            existing->struct_type_name = strdup(sym.struct_type_name);
            existing->type = sym.type;
            existing->element_type = sym.element_type;
            existing->type_info = sym.type_info;
            existing->is_mut = sym.is_mut;
            existing->value = sym.value;
            /* Don't add a new symbol - we updated the existing one */
            return;
        }
    }
    
    /* GC refcount fix: If this string value is already referenced by another
     * variable in the environment, increment refcount to prevent use-after-free
     * when one variable is later reassigned. Handles: let s = (func); let mut r = s */
    if (value.type == VAL_STRING && value.as.string_val && gc_is_managed(value.as.string_val)) {
        for (int i = 0; i < env->symbol_count; i++) {
            if (env->symbols[i].value.type == VAL_STRING &&
                env->symbols[i].value.as.string_val == value.as.string_val) {
                gc_retain(value.as.string_val);
                break;
            }
        }
    }

    env->symbols[env->symbol_count++] = sym;
}

/* Get variable */
Symbol *env_get_var(Environment *env, const char *name) {
    return symbol_lookup(env, name, false);
}

void env_set_current_file(Environment *env, const char *path) {
    if (env) env->current_file = path;
}

const char *env_current_file(Environment *env) {
    return env ? env->current_file : NULL;
}

Symbol *env_get_var_visible_at(Environment *env, const char *name, int line, int column) {
    if (!env || !name) return NULL;
    if (line <= 0) return env_get_var(env, name);

    /* Pick the most recently-defined symbol that is "visible" at this source location.
     * "Visible" here is a best-effort approximation used by later compilation stages
     * (e.g., transpilation) to avoid picking locals from later functions.
     */
    /* Prefer symbols that have a real source location (def_line > 0).
     * Some symbols are inserted without locations (def_line == 0), and if we keep all
     * function-locals across the compilation unit, those "unknown location" symbols can
     * incorrectly shadow well-scoped locals in earlier functions.
     */
    Symbol *best_unknown = NULL;

    /* Pass 1: from most-recent to oldest, return first visible symbol WITH a source location. */
    for (int i = env->symbol_count - 1; i >= 0; i--) {
        Symbol *sym = &env->symbols[i];
        if (!sym->name) continue;
        if (safe_strcmp(sym->name, name) != 0) continue;

        int sline = sym->flow_start_line > 0 ? sym->flow_start_line : sym->def_line;
        int scol = sym->flow_start_line > 0 ? sym->flow_start_column : sym->def_column;
        if (sline <= 0) {
            continue;
        }

        /* Line numbers only mean something inside one file. Comparing a
         * position in the file being processed against a definition in some
         * other module returns whichever unrelated symbol happens to sit at a
         * lower line there -- which is how an imported function's parameter
         * `a` picked up an `a` from the main program and inherited its type,
         * silently lowering float arithmetic to integer opcodes. A symbol from
         * another file is not visible here at all; one with no file recorded
         * still reaches the fallback below, which is what keeps builtins and
         * anything registered without a location working. */
        if (sym->def_file && env->current_file
                && strcmp(sym->def_file, env->current_file) != 0) {
            continue;
        }

        if (sline > line) continue;
        if (sline == line && column > 0 && scol > column) continue;
        if (sym->scope_end_line > 0 &&
            (line > sym->scope_end_line ||
             (line == sym->scope_end_line && column >= sym->scope_end_column))) continue;

        return sym;
    }

    /* Pass 2: fall back to most-recent symbol without a source location. */
    for (int i = env->symbol_count - 1; i >= 0; i--) {
        Symbol *sym = &env->symbols[i];
        if (!sym->name) continue;
        if (safe_strcmp(sym->name, name) != 0) continue;

        if (sym->def_line > 0) continue;
        if (sym->scope_end_line > 0 &&
            (line > sym->scope_end_line ||
             (line == sym->scope_end_line && column >= sym->scope_end_column))) continue;

        best_unknown = sym;
        break;
    }

    return best_unknown;
}

/* Set variable value */
void env_set_var(Environment *env, const char *name, Value value) {
    Symbol *sym = env_get_var(env, name);
    if (sym) {
        /* I copy before releasing the old binding, including self-assignment
         * and a record field borrowed from that binding. */
        if (value.type == VAL_STRUCT && value.as.struct_val) {
            StructValue *sv = value.as.struct_val;
            value = create_struct(sv->struct_name, sv->field_names, sv->field_values, sv->field_count);
        }
        env_free_value(sym->value);
        sym->value = value;

        /* GC refcount fix: If the new string value is already referenced by
         * another variable, increment refcount to prevent use-after-free.
         * Handles: set x y where y is a string variable */
        if (value.type == VAL_STRING && value.as.string_val && gc_is_managed(value.as.string_val)) {
            for (int i = 0; i < env->symbol_count; i++) {
                if (&env->symbols[i] != sym &&
                    env->symbols[i].value.type == VAL_STRING &&
                    env->symbols[i].value.as.string_val == value.as.string_val) {
                    gc_retain(value.as.string_val);
                    break;
                }
            }
        }
    }
}

/* Check if a name is a built-in function (known to the typechecker) */
bool is_builtin_function(const char *name) {
    if (!name) {
        return false;
    }
    const BuiltinEntry *e = builtin_find(name);
    return e && (e->flags & BUILTIN_LANG);
}

/* Helper function to check if a string is in a list */
static bool module_string_list_contains(char **items, int count, const char *name) {
    if (!items || count <= 0 || !name) return false;
    for (int i = 0; i < count; i++) {
        if (items[i] && strcmp(items[i], name) == 0) {
            return true;
        }
    }
    return false;
}

/* I index function slots, never borrowed names or reallocatable pointers.
 * Name-changing external writes invalidate explicitly. Body/module changes are
 * read from the authoritative slot on every lookup. */
struct EnvFunctionIndex {
    int *heads;
    EnvSymbolLink *links;
    size_t bucket_count, capacity;
    int count;
};

void env_function_index_invalidate(Environment *env) {
    if (!env || !env->function_index) return;
    free(env->function_index->heads);
    free(env->function_index->links);
    free(env->function_index);
    env->function_index = NULL;
}

static struct EnvFunctionIndex *function_index_sync(Environment *env) {
    struct EnvFunctionIndex *index = env->function_index;
    if (!index) {
        index = calloc(1, sizeof *index);
        if (!index) return NULL;
        env->function_index = index;
    }
    while (index->count > env->function_count) {
        EnvSymbolLink *link = &index->links[--index->count];
        if (link->previous >= 0)
            index->heads[link->hash & (index->bucket_count - 1)] = link->previous;
    }
    size_t needed = (size_t)env->function_count;
    if (needed > index->capacity) {
        size_t capacity = index->capacity ? index->capacity : 16;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2) goto unavailable;
            capacity *= 2;
        }
        if (capacity > SIZE_MAX / sizeof *index->links) goto unavailable;
        EnvSymbolLink *links = realloc(index->links, capacity * sizeof *links);
        if (!links) goto unavailable;
        index->links = links;
        index->capacity = capacity;
    }
    if (!index->bucket_count || needed > index->bucket_count / 2) {
        size_t buckets = index->bucket_count ? index->bucket_count : 32;
        while (needed > buckets / 2) {
            if (buckets > SIZE_MAX / 2) goto unavailable;
            buckets *= 2;
        }
        if (buckets > SIZE_MAX / sizeof *index->heads) goto unavailable;
        int *heads = calloc(buckets, sizeof *heads);
        if (!heads) goto unavailable;
        for (int i = 0; i < index->count; ++i) {
            EnvSymbolLink *link = &index->links[i];
            if (link->previous < 0) continue;
            size_t bucket = link->hash & (buckets - 1);
            link->previous = heads[bucket];
            heads[bucket] = i + 1;
        }
        free(index->heads);
        index->heads = heads;
        index->bucket_count = buckets;
    }
    while (index->count < env->function_count) {
        int slot = index->count++;
        EnvSymbolLink *link = &index->links[slot];
        const char *name = env->functions[slot].name;
        link->previous = -1;
        if (!name) continue;
        link->hash = symbol_name_hash(name);
        size_t bucket = link->hash & (index->bucket_count - 1);
        link->previous = index->heads[bucket];
        index->heads[bucket] = slot + 1;
    }
    return index;

unavailable:
    /* I retain correct lookup when an optional index allocation fails. */
    env_function_index_invalidate(env);
    return NULL;
}

static Function *function_lookup(Environment *env, const char *name,
                                 const char *module, bool require_module,
                                 bool require_body) {
    struct EnvFunctionIndex *index = function_index_sync(env);
    uint64_t hash = symbol_name_hash(name);
    int next = index ? index->heads[hash & (index->bucket_count - 1)] : env->function_count;
    Function *first = NULL;
    while (next) {
        int slot = next - 1;
        next = index ? index->links[slot].previous : slot;
        if (index && index->links[slot].hash != hash) continue;
        Function *function = &env->functions[slot];
        if (!function->name || safe_strcmp(function->name, name) != 0) continue;
        if (require_body && (function->is_extern || !function->body)) continue;
        if (require_module && !((!module && !function->module_name) ||
            (module && function->module_name && strcmp(module, function->module_name) == 0))) continue;
        /* Bucket chains descend by slot. I retain the first declaration. */
        first = function;
    }
    return first;
}

/* Define function */
void env_define_function(Environment *env, Function func) {
    if (env->function_count >= env->function_capacity) {
        env->function_capacity *= 2;
        env->functions = realloc(env->functions, sizeof(Function) * env->function_capacity);
        if (!env->functions) {
            fprintf(stderr, "Error: Out of memory reallocating functions array\n");
            exit(1);
        }
    }

    env->functions[env->function_count++] = func;
    
    /* Add to module's exported functions list if public and module exists */
    if (func.is_pub && func.module_name) {
        ModuleInfo *mod = env_get_module(env, func.module_name);
        if (mod) {
            /* Check if function is already in the list (avoid duplicates) */
            if (!module_string_list_contains(mod->exported_functions, mod->function_count, func.name)) {
                /* Grow array to accommodate one more element */
                mod->exported_functions = realloc(mod->exported_functions, sizeof(char*) * (mod->function_count + 1));
                mod->exported_functions[mod->function_count++] = strdup(func.name);
            }
        }
    }
}

static bool namespace_owned_by(const ModuleNamespace *ns, const char *owner) {
    return (!ns->owner_module && !owner) ||
           (ns->owner_module && owner && strcmp(ns->owner_module, owner) == 0);
}

/* Get function */
Function *env_get_function(Environment *env, const char *name) {
    if (!name) {
        return NULL;
    }

    /* Check for Module.function pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *func_name = dot + 1;
        
        /* Find namespace */
        for (int i = 0; i < env->namespace_count; i++) {
            if (namespace_owned_by(&env->namespaces[i], env->current_module) &&
                strcmp(env->namespaces[i].alias, module_alias) == 0) {
                /* Check if function is in this namespace */
                for (int j = 0; j < env->namespaces[i].function_count; j++) {
                    if (strcmp(env->namespaces[i].function_names[j], func_name) == 0) {
                        /* Look up the actual function by its original name AND module name */
                        const char *orig_mod = env->namespaces[i].module_name;
                        return function_lookup(env, func_name, orig_mod, true, false);
                    }
                }
                /* Function not found in this module's namespace */
                return NULL;
            }
        }
        /* Module alias not found */
        return NULL;
    }

    /* I permit this non-reserved declaration only in its own module. */
    if (strcmp(name, "array_push") == 0) {
        Function *local = function_lookup(env, name, env->current_module, true, true);
        if (local) return local;
    }

    /* Check built-in functions via unified registry (only BUILTIN_LANG entries) */
    for (int i = 0; i < builtin_registry_count; i++) {
        if (!(builtin_registry[i].flags & BUILTIN_LANG)) continue;
        if (safe_strcmp(builtin_registry[i].name, name) == 0) {
            /* Create static function objects for built-ins */
            static Function func_cache[256];
            static bool initialized[256] = {false};

            if (!initialized[i]) {
                func_cache[i].name = (char *)builtin_registry[i].name;
                func_cache[i].param_count = builtin_registry[i].arity;
                func_cache[i].return_type = builtin_registry[i].return_type;
                func_cache[i].params = NULL;  /* Built-ins don't need param names */
                func_cache[i].body = NULL;
                func_cache[i].shadow_test = NULL;
                initialized[i] = true;
            }

            return &func_cache[i];
        }
    }

    Function *local = function_lookup(env, name, env->current_module, true, false);
    return local ? local : function_lookup(env, name, NULL, false, false);
}

/* I share push identity across inference and native lowering. */
bool env_array_push_is_builtin(Environment *env, int line, int column) {
    if (env_get_var_visible_at(env, "array_push", line, column)) return false;
    Function *function = env_get_function(env, "array_push");
    return !function || !function->body;
}

/* Value creation functions */
Value create_int(long long val) {
    Value v;
    v.type = VAL_INT;
    v.is_return = false;
    v.is_break = false;
    v.is_continue = false;
    v.as.int_val = val;
    return v;
}

Value create_float(double val) {
    Value v;
    v.type = VAL_FLOAT;
    v.is_return = false;
    v.is_break = false;
    v.is_continue = false;
    v.as.float_val = val;
    return v;
}

Value create_bool(bool val) {
    Value v;
    v.type = VAL_BOOL;
    v.is_return = false;
    v.is_break = false;
    v.is_continue = false;
    v.as.bool_val = val;
    return v;
}

Value create_string(const char *val) {
    Value v;
    v.type = VAL_STRING;
    v.is_return = false;
    v.is_break = false;
    v.is_continue = false;

    /* Use GC allocation for all dynamically created strings */
    size_t len = strlen(val);
    char *gc_str = gc_alloc_string(len);
    if (gc_str) {
        memcpy(gc_str, val, len);
        gc_str[len] = '\0';
        v.as.string_val = gc_str;
    } else {
        /* Fallback to empty string on allocation failure */
        v.as.string_val = gc_alloc_string(0);
    }

    return v;
}

Value create_void(void) {
    Value v;
    v.type = VAL_VOID;
    v.is_return = false;
    v.is_break = false;
    v.is_continue = false;
    return v;
}

Value create_array(ValueType elem_type, int length, int capacity) {
    Value v;
    v.type = VAL_ARRAY;
    v.is_return = false;
    v.is_break = false;
    v.is_continue = false;
    v.as.array_val = malloc(sizeof(Array));
    v.as.array_val->element_type = elem_type;
    v.as.array_val->length = length;
    v.as.array_val->capacity = capacity > length ? capacity : length;
    
    /* Allocate data based on element type */
    size_t elem_size;
    switch (elem_type) {
        case VAL_INT:    elem_size = sizeof(long long); break;
        case VAL_FLOAT:  elem_size = sizeof(double); break;
        case VAL_BOOL:   elem_size = sizeof(bool); break;
        case VAL_STRING: elem_size = sizeof(char*); break;
        case VAL_ARRAY:  elem_size = sizeof(Value); break;
        default:         elem_size = sizeof(void*); break;
    }
    v.as.array_val->data = calloc(v.as.array_val->capacity, elem_size);
    
    return v;
}

Value create_struct(const char *struct_name, char **field_names, Value *field_values, int field_count) {
    Value v;
    v.type = VAL_STRUCT;
    v.is_return = false;
    v.is_break = false;
    v.is_continue = false;
    v.as.struct_val = malloc(sizeof(StructValue));
    v.as.struct_val->struct_name = strdup(struct_name);
    v.as.struct_val->field_count = field_count;
    
    /* Allocate and copy field names */
    v.as.struct_val->field_names = malloc(sizeof(char*) * field_count);
    for (int i = 0; i < field_count; i++) {
        v.as.struct_val->field_names[i] = strdup(field_names[i]);
    }
    
    /* Allocate and copy field values */
    v.as.struct_val->field_values = malloc(sizeof(Value) * field_count);
    for (int i = 0; i < field_count; i++) {
        if (field_values[i].type == VAL_STRING) {
            const char *src = field_values[i].as.string_val ? field_values[i].as.string_val : "";
            v.as.struct_val->field_values[i] = create_string(src);
        } else if (field_values[i].type == VAL_STRUCT && field_values[i].as.struct_val) {
            StructValue *nested = field_values[i].as.struct_val;
            v.as.struct_val->field_values[i] = create_struct(nested->struct_name,
                nested->field_names, nested->field_values, nested->field_count);
        } else {
            v.as.struct_val->field_values[i] = field_values[i];
        }
    }
    
    return v;
}

Value create_union(const char *union_name, int variant_index, const char *variant_name, 
                   char **field_names, Value *field_values, int field_count) {
    Value v;
    v.type = VAL_UNION;
    v.is_return = false;
    v.is_break = false;
    v.is_continue = false;
    v.as.union_val = malloc(sizeof(UnionValue));
    v.as.union_val->union_name = strdup(union_name);
    v.as.union_val->variant_index = variant_index;
    v.as.union_val->variant_name = strdup(variant_name);
    v.as.union_val->field_count = field_count;
    
    /* Allocate and copy field names */
    if (field_count > 0) {
        v.as.union_val->field_names = malloc(sizeof(char*) * field_count);
        for (int i = 0; i < field_count; i++) {
            v.as.union_val->field_names[i] = strdup(field_names[i]);
        }
        
        /* Allocate and copy field values */
        v.as.union_val->field_values = malloc(sizeof(Value) * field_count);
        for (int i = 0; i < field_count; i++) {
            /* My payload outlives the constructing function's local strings. */
            if (field_values[i].type == VAL_STRING) {
                v.as.union_val->field_values[i] = create_string(
                    field_values[i].as.string_val ? field_values[i].as.string_val : "");
            } else if (field_values[i].type == VAL_STRUCT && field_values[i].as.struct_val) {
                StructValue *nested = field_values[i].as.struct_val;
                v.as.union_val->field_values[i] = create_struct(nested->struct_name,
                    nested->field_names, nested->field_values, nested->field_count);
            } else {
                v.as.union_val->field_values[i] = field_values[i];
            }
        }
    } else {
        v.as.union_val->field_names = NULL;
        v.as.union_val->field_values = NULL;
    }
    
    return v;
}

/* I test declaration identity without importing another module's fallback. */
StructDef *env_get_struct_owned(Environment *env, const char *name, const char *owner) {
    if (!env || !name) return NULL;
    for (int i = 0; i < env->struct_count; ++i) {
        StructDef *record = &env->structs[i];
        bool same_owner = (!owner && !record->module_name) ||
            (owner && record->module_name && strcmp(owner, record->module_name) == 0);
        if (same_owner && record->name && (strcmp(name, record->name) == 0 || (record->original_name && strcmp(name, record->original_name) == 0))) return record;
    }
    return NULL;
}

/* Define struct */
void env_define_struct(Environment *env, StructDef struct_def) {
    /* Check if struct already exists - prevent duplicates */
    if (env_get_struct_owned(env, struct_def.name, struct_def.module_name) != NULL) {
        /* Struct already defined - skip duplicate registration */
        return;
    }
    
    if (env->struct_count >= env->struct_capacity) {
        env->struct_capacity *= 2;
        env->structs = realloc(env->structs, sizeof(StructDef) * env->struct_capacity);
    }
    env->structs[env->struct_count++] = struct_def;

    /* Module introspection: track exported structs (public only).
     * NOTE: Use the centralized helper to avoid mismatched allocation strategies.
     */
    if (struct_def.is_pub && struct_def.module_name) {
        env_add_module_exported_struct(env, struct_def.module_name, struct_def.original_name ? struct_def.original_name : struct_def.name);
    }

}

/* Get struct definition */
StructDef *env_get_struct(Environment *env, const char *name) {
    /* Check for Module.Type pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *type_name = dot + 1;
        
        /* Find namespace */
        for (int i = 0; i < env->namespace_count; i++) {
            if (namespace_owned_by(&env->namespaces[i], env->current_module) &&
                strcmp(env->namespaces[i].alias, module_alias) == 0) {
                /* Check if struct is in this namespace */
                for (int j = 0; j < env->namespaces[i].struct_count; j++) {
                    if (strcmp(env->namespaces[i].struct_names[j], type_name) == 0) {
                        /* Look up the actual struct by its original name AND module name */
                        const char *orig_mod = env->namespaces[i].module_name;
                        for (int k = 0; k < env->struct_count; k++) {
                            if ((safe_strcmp(env->structs[k].name, type_name) == 0 || (env->structs[k].original_name && strcmp(env->structs[k].original_name, type_name) == 0))) {
                                if (!orig_mod || !env->structs[k].module_name ||
                                    strcmp(env->structs[k].module_name, orig_mod) == 0) {
                                    return &env->structs[k];
                                }
                            }
                        }
                    }
                }
                return NULL;
            }
        }
        return NULL;
    }
    
    /* First pass: prefer structs in the current module */
    if (env->current_module) {
        for (int i = 0; i < env->struct_count; i++) {
            if (env->structs[i].name && (safe_strcmp(env->structs[i].name, name) == 0 || (env->structs[i].original_name && strcmp(env->structs[i].original_name, name) == 0))) {
                if (env->structs[i].module_name && 
                    strcmp(env->structs[i].module_name, env->current_module) == 0) {
                    return &env->structs[i];
                }
            }
        }
    }

    for (int i = 0; i < env->struct_count; i++) {
        if ((safe_strcmp(env->structs[i].name, name) == 0 || (env->structs[i].original_name && strcmp(env->structs[i].original_name, name) == 0))) {
            return &env->structs[i];
        }
    }
    return NULL;
}

/* Define enum */
void env_define_enum(Environment *env, EnumDef enum_def) {
    if (!env || !enum_def.name) {
        return;  /* Invalid enum definition */
    }
    
    /* Check if enum already exists - prevent duplicates */
    if (env_get_enum(env, enum_def.name) != NULL) {
        /* Enum already defined - skip duplicate registration */
        return;
    }
    
    if (env->enum_count >= env->enum_capacity) {
        int old_capacity = env->enum_capacity;
        env->enum_capacity *= 2;
        EnumDef *new_enums = realloc(env->enums, sizeof(EnumDef) * env->enum_capacity);
        if (!new_enums) {
            fprintf(stderr, "Error: Failed to reallocate memory for enums\n");
            return;
        }
        env->enums = new_enums;
        /* Zero-initialize the newly allocated memory */
        memset(&env->enums[old_capacity], 0, sizeof(EnumDef) * (env->enum_capacity - old_capacity));
    }
    env->enums[env->enum_count++] = enum_def;
}

/* Get enum definition */
EnumDef *env_get_enum(Environment *env, const char *name) {
    if (!env || !name) return NULL;
    
    if (!env->enums) return NULL;
    
    /* Defensive check: ensure enum_count is valid */
    if (env->enum_count < 0 || env->enum_count > env->enum_capacity) {
        return NULL;
    }
    
    /* Check for Module.Type pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *type_name = dot + 1;
        
        for (int i = 0; i < env->namespace_count; i++) {
            if (namespace_owned_by(&env->namespaces[i], env->current_module) &&
                strcmp(env->namespaces[i].alias, module_alias) == 0) {
                for (int j = 0; j < env->namespaces[i].enum_count; j++) {
                    if (strcmp(env->namespaces[i].enum_names[j], type_name) == 0) {
                        /* Look up the actual enum by its original name AND module name */
                        const char *orig_mod = env->namespaces[i].module_name;
                        for (int k = 0; k < env->enum_count; k++) {
                            if (safe_strcmp(env->enums[k].name, type_name) == 0) {
                                if (!orig_mod || !env->enums[k].module_name ||
                                    strcmp(env->enums[k].module_name, orig_mod) == 0) {
                                    return &env->enums[k];
                                }
                            }
                        }
                    }
                }
                return NULL;
            }
        }
        return NULL;
    }
    
    /* First pass: prefer enums in the current module */
    if (env->current_module) {
        for (int i = 0; i < env->enum_count; i++) {
            if (env->enums[i].name && safe_strcmp(env->enums[i].name, name) == 0) {
                if (env->enums[i].module_name && 
                    strcmp(env->enums[i].module_name, env->current_module) == 0) {
                    return &env->enums[i];
                }
            }
        }
    }

    for (int i = 0; i < env->enum_count; i++) {
        /* Use safe_strcmp which handles NULL pointers */
        if (safe_strcmp(env->enums[i].name, name) == 0) {
            return &env->enums[i];
        }
    }
    return NULL;
}

/* Get enum variant value */
int env_get_enum_variant(Environment *env, const char *variant_name) {
    if (!env || !variant_name) return -1;
    
    for (int i = 0; i < env->enum_count; i++) {
        if (!env->enums[i].variant_names) continue;
        for (int j = 0; j < env->enums[i].variant_count; j++) {
            if (safe_strcmp(env->enums[i].variant_names[j], variant_name) == 0) {
                return (env->enums[i].variant_values && j < env->enums[i].variant_count) ? 
                    env->enums[i].variant_values[j] : j;
            }
        }
    }
    return -1;  /* Not found */
}

/* Define union */
void env_define_union(Environment *env, UnionDef union_def) {
    /* Check if union already exists - prevent duplicates */
    if (env_get_union(env, union_def.name) != NULL) {
        /* Union already defined - skip duplicate registration */
        return;
    }
    
    if (env->union_count >= env->union_capacity) {
        env->union_capacity *= 2;
        env->unions = realloc(env->unions, sizeof(UnionDef) * env->union_capacity);
    }
    env->unions[env->union_count++] = union_def;
}

/* Get union definition */
UnionDef *env_get_union(Environment *env, const char *name) {
    /* Check for Module.Type pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *type_name = dot + 1;
        
        for (int i = 0; i < env->namespace_count; i++) {
            if (namespace_owned_by(&env->namespaces[i], env->current_module) &&
                strcmp(env->namespaces[i].alias, module_alias) == 0) {
                for (int j = 0; j < env->namespaces[i].union_count; j++) {
                    if (strcmp(env->namespaces[i].union_names[j], type_name) == 0) {
                        /* Look up the actual union by its original name AND module name */
                        const char *orig_mod = env->namespaces[i].module_name;
                        for (int k = 0; k < env->union_count; k++) {
                            if (safe_strcmp(env->unions[k].name, type_name) == 0) {
                                if (!orig_mod || !env->unions[k].module_name ||
                                    strcmp(env->unions[k].module_name, orig_mod) == 0) {
                                    return &env->unions[k];
                                }
                            }
                        }
                    }
                }
                return NULL;
            }
        }
        return NULL;
    }
    
    /* First pass: prefer unions in the current module */
    if (env->current_module) {
        for (int i = 0; i < env->union_count; i++) {
            if (env->unions[i].name && safe_strcmp(env->unions[i].name, name) == 0) {
                if (env->unions[i].module_name && 
                    strcmp(env->unions[i].module_name, env->current_module) == 0) {
                    return &env->unions[i];
                }
            }
        }
    }

    for (int i = 0; i < env->union_count; i++) {
        if (safe_strcmp(env->unions[i].name, name) == 0) {
            return &env->unions[i];
        }
    }
    return NULL;
}

/* Get variant index in union (returns -1 if not found) */
int env_get_union_variant_index(Environment *env, const char *union_name, const char *variant_name) {
    UnionDef *udef = env_get_union(env, union_name);
    if (!udef) return -1;
    
    for (int i = 0; i < udef->variant_count; i++) {
        if (safe_strcmp(udef->variant_names[i], variant_name) == 0) {
            return i;
        }
    }
    return -1;
}

/* Define opaque type */
void env_define_opaque_type(Environment *env, const char *name) {
    /* Check if opaque type already exists - prevent duplicates */
    if (env_get_opaque_type(env, name) != NULL) {
        /* Opaque type already defined - skip duplicate registration */
        return;
    }
    
    if (env->opaque_type_count >= env->opaque_type_capacity) {
        env->opaque_type_capacity *= 2;
        env->opaque_types = realloc(env->opaque_types, sizeof(OpaqueTypeDef) * env->opaque_type_capacity);
    }
    
    OpaqueTypeDef opaque_type;
    opaque_type.name = strdup(name);
    
    /* Generate C type name by adding pointer: "GLFWwindow" -> "GLFWwindow*" */
    size_t len = strlen(name);
    opaque_type.c_type_name = malloc(len + 2);  /* +1 for '*', +1 for '\0' */
    strcpy(opaque_type.c_type_name, name);
    strcat(opaque_type.c_type_name, "*");
    
    env->opaque_types[env->opaque_type_count++] = opaque_type;
}

/* Get opaque type definition */
OpaqueTypeDef *env_get_opaque_type(Environment *env, const char *name) {
    if (!env || !name) return NULL;
    
    /* Check for Module.Type pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *type_name = dot + 1;
        
        /* Opaque types aren't explicitly tracked in namespaces yet, 
         * but we can still search for them globally with module matching if we add it.
         * For now, just search globally by short name as a fallback.
         */
        return env_get_opaque_type(env, type_name);
    }
    
    for (int i = 0; i < env->opaque_type_count; i++) {
        if (safe_strcmp(env->opaque_types[i].name, name) == 0) {
            return &env->opaque_types[i];
        }
    }
    return NULL;
}

/* Register a list instantiation for code generation */
void env_register_list_instantiation(Environment *env, const char *element_type) {
    /* Check if already registered */
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (safe_strcmp(inst->generic_name, "List") == 0 &&
            inst->type_arg_names && 
            safe_strcmp(inst->type_arg_names[0], element_type) == 0) {
            return;  /* Already registered */
        }
    }
    
    /* Add new instantiation */
    if (env->generic_instance_count >= env->generic_instance_capacity) {
        env->generic_instance_capacity *= 2;
        env->generic_instances = realloc(env->generic_instances,
            sizeof(GenericInstantiation) * env->generic_instance_capacity);
    }
    
    GenericInstantiation inst = {0};
    inst.generic_name = strdup("List");
    inst.type_arg_count = 1;
    inst.type_args = malloc(sizeof(Type));
    inst.type_args[0] = TYPE_LIST_GENERIC;
    inst.type_arg_names = malloc(sizeof(char*));
    inst.type_arg_names[0] = strdup(element_type);
    
    /* Generate specialized name: List<Point> -> List_Point */
    char specialized[256];
    snprintf(specialized, sizeof(specialized), "List_%s", element_type);
    inst.concrete_name = strdup(specialized);
    
    env->generic_instances[env->generic_instance_count++] = inst;
    
    /* Register specialized functions in environment for type checking */
    char func_name[512];  /* Increased to handle long type names + suffixes */
    /* Important: zero-init so module/visibility pointers don't contain garbage.
     * These generated externs are treated like builtins during typechecking. */
    /* I own generated declaration storage independently of function slots;
     * ordinary declarations continue to borrow their AST metadata. */
    Function func = (Function){0};
    Parameter *params;

    func.is_extern = true;
    func.is_pub = false;
    func.module_name = NULL;
    
    /* List_T_new() -> List<T>* */
    snprintf(func_name, sizeof(func_name), "%s_new", specialized);
    func.name = env_own_checker_allocation(env, strdup(func_name));
    func.param_count = 0;
    func.params = NULL;
    func.return_type = TYPE_LIST_GENERIC;
    func.return_struct_type_name = NULL;
    func.return_fn_sig = NULL;
    func.return_type_info = NULL;
    func.body = NULL;  /* Built-in */
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* List_T_push(list: List<T>*, value: T) -> void */
    snprintf(func_name, sizeof(func_name), "%s_push", specialized);
    func.name = env_own_checker_allocation(env, strdup(func_name));
    func.param_count = 2;
    params = env_own_checker_allocation(env, calloc(2, sizeof(Parameter)));
    params[0].name = env_own_checker_allocation(env, strdup("list"));
    params[0].type = TYPE_LIST_GENERIC;
    params[0].struct_type_name = NULL;
    params[0].element_type = TYPE_UNKNOWN;
    params[1].name = env_own_checker_allocation(env, strdup("value"));
    params[1].type = TYPE_STRUCT;
    params[1].struct_type_name = env_own_checker_allocation(env, strdup(element_type));
    params[1].element_type = TYPE_UNKNOWN;
    func.params = params;
    func.return_type = TYPE_VOID;
    func.return_struct_type_name = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = true;
    env_define_function(env, func);
    
    /* List_T_get(list: List<T>*, index: int) -> T */
    snprintf(func_name, sizeof(func_name), "%s_get", specialized);
    func.name = env_own_checker_allocation(env, strdup(func_name));
    func.param_count = 2;
    params = env_own_checker_allocation(env, calloc(2, sizeof(Parameter)));
    params[0].name = env_own_checker_allocation(env, strdup("list"));
    params[0].type = TYPE_LIST_GENERIC;
    params[0].struct_type_name = NULL;
    params[0].element_type = TYPE_UNKNOWN;
    params[1].name = env_own_checker_allocation(env, strdup("index"));
    params[1].type = TYPE_INT;
    params[1].struct_type_name = NULL;
    params[1].element_type = TYPE_UNKNOWN;
    func.params = params;
    func.return_type = TYPE_STRUCT;
    func.return_struct_type_name = env_own_checker_allocation(env, strdup(element_type));
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = true;
    env_define_function(env, func);
    
    /* List_T_length(list: List<T>*) -> int */
    snprintf(func_name, sizeof(func_name), "%s_length", specialized);
    func.name = env_own_checker_allocation(env, strdup(func_name));
    func.param_count = 1;
    params = env_own_checker_allocation(env, calloc(1, sizeof(Parameter)));
    params[0].name = env_own_checker_allocation(env, strdup("list"));
    params[0].type = TYPE_LIST_GENERIC;
    params[0].struct_type_name = NULL;
    params[0].element_type = TYPE_UNKNOWN;
    func.params = params;
    func.return_type = TYPE_INT;
    func.return_struct_type_name = NULL;
    func.return_fn_sig = NULL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = true;
    env_define_function(env, func);
}

/* Register a HashMap<K,V> instantiation for code generation
 * Example: HashMap<string, int> -> HashMap_string_int
 */
void env_register_hashmap_instantiation(Environment *env, const char *key_type, const char *value_type) {
    if (!env || !key_type || !value_type) return;

    /* Check if already registered */
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (safe_strcmp(inst->generic_name, "HashMap") == 0 &&
            inst->type_arg_count == 2 && inst->type_arg_names &&
            safe_strcmp(inst->type_arg_names[0], key_type) == 0 &&
            safe_strcmp(inst->type_arg_names[1], value_type) == 0) {
            return;
        }
    }

    if (env->generic_instance_count >= env->generic_instance_capacity) {
        env->generic_instance_capacity *= 2;
        env->generic_instances = realloc(env->generic_instances,
            sizeof(GenericInstantiation) * env->generic_instance_capacity);
    }

    GenericInstantiation inst = {0};
    inst.generic_name = strdup("HashMap");
    inst.type_arg_count = 2;
    inst.type_args = malloc(sizeof(Type) * 2);
    inst.type_arg_names = malloc(sizeof(char*) * 2);
    inst.type_args[0] = TYPE_UNKNOWN;
    inst.type_args[1] = TYPE_UNKNOWN;
    inst.type_arg_names[0] = strdup(key_type);
    inst.type_arg_names[1] = strdup(value_type);

    char specialized[512];
    snprintf(specialized, sizeof(specialized), "HashMap_%s_%s", key_type, value_type);
    inst.concrete_name = strdup(specialized);

    env->generic_instances[env->generic_instance_count++] = inst;
}

/* Register a generic union instantiation for code generation
 * Example: Result<int, string> -> Result_int_string
 */
void env_register_union_instantiation(Environment *env, const char *union_name, 
                                     const char **type_args, int type_arg_count) {
    if (!env || !union_name || !type_args || type_arg_count == 0) {
        return;
    }
    
    /* Check if already registered */
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (safe_strcmp(inst->generic_name, union_name) == 0 &&
            inst->type_arg_count == type_arg_count) {
            /* Check if all type args match */
            bool all_match = true;
            for (int j = 0; j < type_arg_count; j++) {
                if (safe_strcmp(inst->type_arg_names[j], type_args[j]) != 0) {
                    all_match = false;
                    break;
                }
            }
            if (all_match) {
                return;  /* Already registered */
            }
        }
    }
    
    /* Add new instantiation */
    if (env->generic_instance_count >= env->generic_instance_capacity) {
        env->generic_instance_capacity *= 2;
        env->generic_instances = realloc(env->generic_instances,
            sizeof(GenericInstantiation) * env->generic_instance_capacity);
    }
    
    GenericInstantiation inst = {0};
    inst.generic_name = strdup(union_name);
    inst.type_arg_count = type_arg_count;
    inst.type_args = malloc(sizeof(Type) * type_arg_count);
    inst.type_arg_names = malloc(sizeof(char*) * type_arg_count);
    
    for (int i = 0; i < type_arg_count; i++) {
        inst.type_args[i] = TYPE_UNION;  /* Generic union type */
        inst.type_arg_names[i] = strdup(type_args[i]);
    }
    
    /* Generate specialized name: Result<int, string> -> Result_int_string */
    char specialized[512];
    int offset = snprintf(specialized, sizeof(specialized), "%s", union_name);
    for (int i = 0; i < type_arg_count && offset < (int)sizeof(specialized) - 20; i++) {
        offset += snprintf(specialized + offset, sizeof(specialized) - offset, 
                          "_%s", type_args[i]);
    }
    inst.concrete_name = strdup(specialized);
    
    env->generic_instances[env->generic_instance_count++] = inst;
}

/* ============================================================================
 * Function Signature Helpers (for first-class functions)
 * ============================================================================
 */

/* Create a function signature */
FunctionSignature *create_function_signature(Type *param_types, int param_count, Type return_type) {
    FunctionSignature *sig = calloc(1, sizeof(FunctionSignature));
    sig->param_count = param_count;
    sig->return_type = return_type;
    sig->return_struct_name = NULL;
    sig->return_fn_sig = NULL;  /* Initialize function return signature */
    
    if (param_count > 0) {
        sig->param_types = malloc(sizeof(Type) * param_count);
        sig->param_struct_names = malloc(sizeof(char*) * param_count);
        
        for (int i = 0; i < param_count; i++) {
            sig->param_types[i] = param_types[i];
            sig->param_struct_names[i] = NULL;
        }
    } else {
        sig->param_types = NULL;
        sig->param_struct_names = NULL;
    }
    
    return sig;
}

/* Free a function signature */
void free_function_signature(FunctionSignature *sig) {
    if (!sig) return;
    
    if (sig->param_types) {
        free(sig->param_types);
    }
    
    if (sig->param_struct_names) {
        for (int i = 0; i < sig->param_count; i++) {
            if (sig->param_struct_names[i]) {
                free(sig->param_struct_names[i]);
            }
        }
        free(sig->param_struct_names);
    }
    
    if (sig->return_struct_name) {
        free(sig->return_struct_name);
    }
    
    for (int i = 0; sig->param_type_info && i < sig->param_count; ++i)
        free_payload_type_info(sig->param_type_info[i]);
    free(sig->param_type_info);
    free_payload_type_info(sig->return_type_info);

    /* Free nested function signature if present */
    if (sig->return_fn_sig) {
        free_function_signature(sig->return_fn_sig);
    }
    
    free(sig);
}

void free_type_info(TypeInfo *info) {
    if (!info) return;

    if (info->element_type) {
        free_type_info(info->element_type);
    }
    if (info->generic_name) {
        free(info->generic_name);
    }
    if (info->type_params) {
        for (int i = 0; i < info->type_param_count; i++) {
            free_type_info(info->type_params[i]);
        }
        free(info->type_params);
    }
    if (info->tuple_types) {
        free(info->tuple_types);
    }
    if (info->tuple_type_names) {
        for (int i = 0; i < info->tuple_element_count; i++) {
            if (info->tuple_type_names[i]) {
                free(info->tuple_type_names[i]);
            }
        }
        free(info->tuple_type_names);
    }
    if (info->opaque_type_name) {
        free(info->opaque_type_name);
    }
    if (info->fn_sig) {
        free_function_signature(info->fn_sig);
    }
    free(info);
}

/* Payload annotations own their complete parsed tree. I never borrow a nested
 * node across AST, environment and extracted-module lifetimes. */
static void *payload_alloc(size_t count, size_t size) {
    if (count && size > SIZE_MAX / count) {
        fprintf(stderr, "I cannot represent this payload type metadata\n");
        exit(1);
    }
    void *value = calloc(count ? count : 1, size);
    if (!value) {
        fprintf(stderr, "I cannot allocate payload type metadata\n");
        exit(1);
    }
    return value;
}
static char *payload_name(const char *name) {
    if (!name) return NULL;
    size_t length = strlen(name) + 1;
    char *copy = payload_alloc(length, 1);
    memcpy(copy, name, length);
    return copy;
}
static char **payload_names(char *const *names, int count) {
    if (!names) return NULL;
    char **copy = payload_alloc((size_t)count, sizeof(*copy));
    for (int i = 0; i < count; ++i) copy[i] = payload_name(names[i]);
    return copy;
}
static void payload_free_names(char **names, int count) {
    if (!names) return;
    for (int i = 0; i < count; ++i) free(names[i]);
    free(names);
}
static TypeInfo *payload_copy(const TypeInfo *source, unsigned depth);
static FunctionSignature *payload_signature(const FunctionSignature *source, unsigned depth) {
    if (!source) return NULL;
    if (depth > 512) {
        fprintf(stderr, "I cannot copy payload signatures beyond my checked depth\n");
        exit(1);
    }
    FunctionSignature *copy = payload_alloc(1, sizeof(*copy));
    *copy = *source;
    copy->param_types = NULL;
    if (source->param_types) {
        copy->param_types = payload_alloc((size_t)source->param_count, sizeof(Type));
        memcpy(copy->param_types, source->param_types, (size_t)source->param_count * sizeof(Type));
    }
    copy->param_struct_names = payload_names(source->param_struct_names, source->param_count);
    copy->return_struct_name = payload_name(source->return_struct_name);
    copy->return_fn_sig = payload_signature(source->return_fn_sig, depth + 1);
    copy->param_type_info = NULL;
    if (source->param_type_info) {
        copy->param_type_info = payload_alloc((size_t)source->param_count, sizeof(TypeInfo*));
        for (int i = 0; i < source->param_count; ++i)
            copy->param_type_info[i] = payload_copy(source->param_type_info[i], depth + 1);
    }
    copy->return_type_info = payload_copy(source->return_type_info, depth + 1);
    return copy;
}
static TypeInfo *payload_copy(const TypeInfo *source, unsigned depth) {
    if (!source) return NULL;
    if (depth > 512) {
        fprintf(stderr, "I cannot copy payload annotations beyond my checked depth\n");
        exit(1);
    }
    TypeInfo *copy = payload_alloc(1, sizeof(*copy));
    *copy = *source;
    copy->generic_name = payload_name(source->generic_name);
    copy->opaque_type_name = payload_name(source->opaque_type_name);
    copy->element_type = payload_copy(source->element_type, depth + 1);
    copy->type_params = NULL;
    if (source->type_params) {
        copy->type_params = payload_alloc((size_t)source->type_param_count, sizeof(TypeInfo *));
        for (int i = 0; i < source->type_param_count; ++i)
            copy->type_params[i] = payload_copy(source->type_params[i], depth + 1);
    }
    copy->tuple_types = NULL;
    if (source->tuple_types) {
        copy->tuple_types = payload_alloc((size_t)source->tuple_element_count, sizeof(Type));
        memcpy(copy->tuple_types, source->tuple_types, (size_t)source->tuple_element_count * sizeof(Type));
    }
    copy->tuple_type_names = payload_names(source->tuple_type_names, source->tuple_element_count);
    copy->fn_sig = payload_signature(source->fn_sig, depth + 1);
    copy->row_var_name = payload_name(source->row_var_name);
    copy->row_field_names = payload_names(source->row_field_names, source->row_field_count);
    copy->row_field_types = NULL;
    if (source->row_field_types) {
        copy->row_field_types = payload_alloc((size_t)source->row_field_count, sizeof(Type));
        memcpy(copy->row_field_types, source->row_field_types, (size_t)source->row_field_count * sizeof(Type));
    }
    copy->row_field_type_names = payload_names(source->row_field_type_names, source->row_field_count);
    copy->type_var_names = payload_names(source->type_var_names, source->type_var_count);
    return copy;
}
TypeInfo *copy_payload_type_info(const TypeInfo *info) { return payload_copy(info, 0); }
void free_payload_type_info(TypeInfo *info) {
    if (!info) return;
    free_payload_type_info(info->element_type);
    for (int i = 0; info->type_params && i < info->type_param_count; ++i)
        free_payload_type_info(info->type_params[i]);
    free(info->type_params);
    free(info->generic_name);
    free(info->opaque_type_name);
    free(info->tuple_types);
    payload_free_names(info->tuple_type_names, info->tuple_element_count);
    free_function_signature(info->fn_sig);
    free(info->row_var_name);
    payload_free_names(info->row_field_names, info->row_field_count);
    free(info->row_field_types);
    payload_free_names(info->row_field_type_names, info->row_field_count);
    payload_free_names(info->type_var_names, info->type_var_count);
    free(info);
}

/* I substitute complete concrete trees, not the flattened field name. */
static void payload_substitute(TypeInfo **slot, const UnionDef *def, const TypeInfo *arguments) {
    TypeInfo *info = *slot;
    if (!info || !arguments) return;
    if (info->generic_name && info->type_param_count == 0 && !info->element_type) {
        for (int i = 0; i < def->generic_param_count && i < arguments->type_param_count; ++i) {
            if (!strcmp(info->generic_name, def->generic_params[i]) && arguments->type_params && arguments->type_params[i]) {
                TypeInfo *concrete = copy_payload_type_info(arguments->type_params[i]);
                free_payload_type_info(info);
                *slot = concrete;
                return;
            }
        }
    }
    if (info->element_type) payload_substitute(&info->element_type, def, arguments);
    for (int i = 0; info->type_params && i < info->type_param_count; ++i)
        payload_substitute(&info->type_params[i], def, arguments);
}
TypeInfo *resolve_union_payload_type_info(const UnionDef *def, int arm, int field, const TypeInfo *arguments) {
    if (!def || arm < 0 || arm >= def->variant_count || field < 0 ||
        field >= def->variant_field_counts[arm] || !def->variant_field_type_info ||
        !def->variant_field_type_info[arm]) return NULL;
    TypeInfo *info = copy_payload_type_info(def->variant_field_type_info[arm][field]);
    payload_substitute(&info, def, arguments);
    return info;
}

FunctionSignature *copy_function_signature(const FunctionSignature *signature) {
    return payload_signature(signature, 0);
}

/* I own every annotation in signatures reconstructed from declarations. */
FunctionSignature *function_signature_from_function(const Function *function) {
    if (!function) return NULL;
    FunctionSignature *sig = payload_alloc(1, sizeof *sig);
    sig->param_count = function->param_count;
    sig->return_type = function->return_type;
    sig->return_struct_name = payload_name(function->return_struct_type_name);
    sig->return_fn_sig = payload_signature(function->return_fn_sig, 0);
    sig->return_type_info = payload_copy(function->return_type_info, 0);
    if (sig->param_count) {
        sig->param_types = payload_alloc((size_t)sig->param_count, sizeof(Type));
        sig->param_struct_names = payload_alloc((size_t)sig->param_count, sizeof(char*));
        sig->param_type_info = payload_alloc((size_t)sig->param_count, sizeof(TypeInfo*));
        for (int i = 0; i < sig->param_count; ++i) {
            const Parameter *param = &function->params[i];
            sig->param_types[i] = param->type;
            sig->param_struct_names[i] = payload_name(param->struct_type_name);
            sig->param_type_info[i] = payload_copy(param->type_info, 0);
        }
    }
    return sig;
}

/* I compare annotation trees, including the legacy flattened tuple/row fields.
 * A recursion limit rejects unresolved cycles instead of accepting a guess. */
static bool annotation_names_equal(const char *left, const char *right) {
    return left == right || (left && right && strcmp(left, right) == 0);
}
static bool signatures_equal_depth(const FunctionSignature *, const FunctionSignature *, unsigned);
static bool annotations_equal(const TypeInfo *a, const TypeInfo *b, unsigned depth) {
    if (a == b) return true;
    if (!a || !b || depth > 128 || a->base_type != b->base_type ||
        a->type_param_count != b->type_param_count ||
        a->tuple_element_count != b->tuple_element_count ||
        a->row_field_count != b->row_field_count || a->type_var_count != b->type_var_count ||
        a->is_open_row != b->is_open_row ||
        !annotation_names_equal(a->generic_name, b->generic_name) ||
        !annotation_names_equal(a->opaque_type_name, b->opaque_type_name) ||
        !annotation_names_equal(a->row_var_name, b->row_var_name)) return false;
    if (!annotations_equal(a->element_type, b->element_type, depth + 1)) return false;
    for (int i = 0; i < a->type_param_count; ++i)
        if (!a->type_params || !b->type_params ||
            !annotations_equal(a->type_params[i], b->type_params[i], depth + 1)) return false;
    for (int i = 0; i < a->tuple_element_count; ++i)
        if (!a->tuple_types || !b->tuple_types || a->tuple_types[i] != b->tuple_types[i] ||
            !annotation_names_equal(a->tuple_type_names ? a->tuple_type_names[i] : NULL,
                                    b->tuple_type_names ? b->tuple_type_names[i] : NULL)) return false;
    for (int i = 0; i < a->row_field_count; ++i)
        if (!a->row_field_types || !b->row_field_types ||
            a->row_field_types[i] != b->row_field_types[i] ||
            !annotation_names_equal(a->row_field_names ? a->row_field_names[i] : NULL,
                                    b->row_field_names ? b->row_field_names[i] : NULL) ||
            !annotation_names_equal(a->row_field_type_names ? a->row_field_type_names[i] : NULL,
                                    b->row_field_type_names ? b->row_field_type_names[i] : NULL)) return false;
    for (int i = 0; i < a->type_var_count; ++i)
        if (!a->type_var_names || !b->type_var_names ||
            !annotation_names_equal(a->type_var_names[i], b->type_var_names[i])) return false;
    return (!a->fn_sig && !b->fn_sig) || signatures_equal_depth(a->fn_sig, b->fn_sig, depth + 1);
}
bool type_infos_equal(const TypeInfo *left, const TypeInfo *right) {
    return annotations_equal(left, right, 0);
}

static bool signature_annotation_equal(Type a, const char *an, const TypeInfo *ai,
                                       Type b, const char *bn, const TypeInfo *bi,
                                       unsigned depth) {
    TypeInfo af = {.base_type = a, .generic_name = (char*)an};
    TypeInfo bf = {.base_type = b, .generic_name = (char*)bn};
    return a == b && annotations_equal(ai ? ai : &af, bi ? bi : &bf, depth + 1);
}
static bool signatures_equal_depth(const FunctionSignature *a, const FunctionSignature *b, unsigned depth) {
    if (!a || !b || depth > 128 || a->param_count != b->param_count) return false;
    for (int i = 0; i < a->param_count; ++i) {
        if (!signature_annotation_equal(a->param_types[i], a->param_struct_names ? a->param_struct_names[i] : NULL,
                a->param_type_info ? a->param_type_info[i] : NULL,
                b->param_types[i], b->param_struct_names ? b->param_struct_names[i] : NULL,
                b->param_type_info ? b->param_type_info[i] : NULL, depth)) return false;
    }
    if (!signature_annotation_equal(a->return_type, a->return_struct_name, a->return_type_info,
                                    b->return_type, b->return_struct_name, b->return_type_info, depth)) return false;
    return a->return_type != TYPE_FUNCTION || signatures_equal_depth(a->return_fn_sig, b->return_fn_sig, depth + 1);
}
bool function_signatures_equal(FunctionSignature *a, FunctionSignature *b) {
    return signatures_equal_depth(a, b, 0);
}

/* Create a function value */
Value create_function(const char *function_name, FunctionSignature *signature) {
    Value val;
    val.type = VAL_FUNCTION;
    val.is_return = false;
    val.is_break = false;
    val.is_continue = false;
    val.as.function_val.function_name = strdup(function_name);
    val.as.function_val.signature = signature;
    return val;
}

/* Create tuple value */
Value create_tuple(Value *elements, int element_count) {
    Value val;
    val.type = VAL_TUPLE;
    val.is_return = false;
    val.is_break = false;
    val.is_continue = false;
    val.as.tuple_val = malloc(sizeof(TupleValue));
    val.as.tuple_val->element_count = element_count;
    
    /* Allocate and copy elements */
    if (element_count > 0) {
        val.as.tuple_val->elements = malloc(sizeof(Value) * element_count);
        for (int i = 0; i < element_count; i++) {
            val.as.tuple_val->elements[i] = elements[i];
            /* Deep copy strings */
            if (elements[i].type == VAL_STRING) {
                val.as.tuple_val->elements[i].as.string_val = strdup(elements[i].as.string_val);
            }
        }
    } else {
        val.as.tuple_val->elements = NULL;
    }
    
    return val;
}

/* Free tuple value */
void free_tuple(TupleValue *tuple) {
    if (!tuple) return;
    
    /* Free string elements */
    for (int i = 0; i < tuple->element_count; i++) {
        if (tuple->elements[i].type == VAL_STRING && tuple->elements[i].as.string_val) {
            free(tuple->elements[i].as.string_val);
        }
    }
    
    if (tuple->elements) {
        free(tuple->elements);
    }
    free(tuple);
}

/* Register a module namespace (for import aliases) */
void env_register_namespace(Environment *env, const char *alias, const char *module_name,
                            char **function_names, int function_count,
                            char **struct_names, int struct_count,
                            char **enum_names, int enum_count,
                            char **union_names, int union_count) {
    if (!env || !alias) {
        return;
    }
    
    /* Check if alias already exists */
    for (int i = 0; i < env->namespace_count; i++) {
        if (namespace_owned_by(&env->namespaces[i], env->current_module) &&
            strcmp(env->namespaces[i].alias, alias) == 0) {
            /* Namespace already registered */
            return;
        }
    }
    
    /* Expand capacity if needed */
    if (env->namespace_count >= env->namespace_capacity) {
        env->namespace_capacity = env->namespace_capacity == 0 ? 4 : env->namespace_capacity * 2;
        env->namespaces = realloc(env->namespaces, sizeof(ModuleNamespace) * env->namespace_capacity);
    }
    
    /* Register the namespace */
    ModuleNamespace *ns = &env->namespaces[env->namespace_count++];
    ns->alias = strdup(alias);
    ns->owner_module = env->current_module ? strdup(env->current_module) : NULL;
    ns->module_name = module_name ? strdup(module_name) : NULL;
    ns->function_names = function_names;
    ns->function_count = function_count;
    ns->struct_names = struct_names;
    ns->struct_count = struct_count;
    ns->enum_names = enum_names;
    ns->enum_count = enum_count;
    ns->union_names = union_names;
    ns->union_count = union_count;
}

/* Module introspection helper functions */

/* Register a module in the environment for introspection */
void env_register_module(Environment *env, const char *name, const char *path, bool is_unsafe) {
    /* Grow module array if needed */
    if (env->module_count >= env->module_capacity) {
        env->module_capacity = env->module_capacity == 0 ? 4 : env->module_capacity * 2;
        env->modules = realloc(env->modules, sizeof(ModuleInfo) * env->module_capacity);
    }
    
    /* Check if module already exists */
    for (int i = 0; i < env->module_count; i++) {
        if (strcmp(env->modules[i].name, name) == 0) {
            /* Module already registered, update is_unsafe flag */
            env->modules[i].is_unsafe = is_unsafe;
            return;
        }
    }
    
    /* Register new module */
    ModuleInfo *mod = &env->modules[env->module_count++];
    mod->name = strdup(name);
    mod->path = path ? strdup(path) : NULL;
    mod->is_unsafe = is_unsafe;
    mod->has_ffi = false;  /* Will be updated during typechecking */
    mod->exported_functions = NULL;
    mod->function_count = 0;
    mod->exported_structs = NULL;
    mod->struct_count = 0;
}

/* Get module info by name */
ModuleInfo *env_get_module(Environment *env, const char *name) {
    for (int i = 0; i < env->module_count; i++) {
        if (strcmp(env->modules[i].name, name) == 0) {
            return &env->modules[i];
        }
    }
    return NULL;
}

/* Check if current module is unsafe */
bool env_is_current_module_unsafe(Environment *env) {
    return env->current_module_is_unsafe;
}

/* Mark module as having FFI (extern functions) */
void env_mark_module_has_ffi(Environment *env, const char *name) {
    ModuleInfo *mod = env_get_module(env, name);
    if (mod) {
        mod->has_ffi = true;
    }
}

void env_add_module_exported_function(Environment *env, const char *module_name, const char *function_name) {
    if (!env || !module_name || !function_name) return;

    ModuleInfo *mod = env_get_module(env, module_name);
    if (!mod) {
        /* Best-effort: ensure the module exists so we can track exports */
        env_register_module(env, module_name, NULL, false);
        mod = env_get_module(env, module_name);
        if (!mod) return;
    }

    if (module_string_list_contains(mod->exported_functions, mod->function_count, function_name)) {
        return;
    }

    mod->exported_functions = realloc(mod->exported_functions, sizeof(char*) * (mod->function_count + 1));
    mod->exported_functions[mod->function_count++] = strdup(function_name);
}

void env_add_module_exported_struct(Environment *env, const char *module_name, const char *struct_name) {
    if (!env || !module_name || !struct_name) return;

    ModuleInfo *mod = env_get_module(env, module_name);
    if (!mod) {
        /* Best-effort: ensure the module exists so we can track exports */
        env_register_module(env, module_name, NULL, false);
        mod = env_get_module(env, module_name);
        if (!mod) return;
    }

    if (module_string_list_contains(mod->exported_structs, mod->struct_count, struct_name)) {
        return;
    }

    mod->exported_structs = realloc(mod->exported_structs, sizeof(char*) * (mod->struct_count + 1));
    mod->exported_structs[mod->struct_count++] = strdup(struct_name);
}

/* ── Algebraic Effects ───────────────────────────────────────────────────── */

void env_define_effect(Environment *env, EffectDef effect_def) {
    if (!env) return;
    /* Prevent duplicates */
    if (env_get_effect(env, effect_def.name) != NULL) return;

    if (env->effect_count >= env->effect_capacity) {
        env->effect_capacity = env->effect_capacity ? env->effect_capacity * 2 : 8;
        env->effects = realloc(env->effects, sizeof(EffectDef) * env->effect_capacity);
    }
    env->effects[env->effect_count++] = effect_def;
}

EffectDef *env_get_effect(Environment *env, const char *name) {
    if (!env || !name) return NULL;
    for (int i = 0; i < env->effect_count; i++) {
        if (env->effects[i].name && strcmp(env->effects[i].name, name) == 0) {
            return &env->effects[i];
        }
    }
    return NULL;
}

EffectOp *effect_get_op(EffectDef *effect, const char *op_name) {
    if (!effect || !op_name) return NULL;
    for (int i = 0; i < effect->op_count; i++) {
        if (effect->ops[i].name && strcmp(effect->ops[i].name, op_name) == 0) {
            return &effect->ops[i];
        }
    }
    return NULL;
}

/* Register a generic function instantiation for monomorphization */
void env_register_generic_func_instance(Environment *env, const char *orig_name, const char *mono_name,
                                         const char **var_names, Type *bound_types, const char **bound_type_names,
                                         int binding_count) {
    if (!env || !orig_name || !mono_name) return;
    /* Check if already registered (avoid duplicates) */
    for (int i = 0; i < env->generic_func_instance_count; i++) {
        if (strcmp(env->generic_func_instances[i].mono_name, mono_name) == 0) return;
    }
    /* Grow if needed */
    if (env->generic_func_instance_count >= env->generic_func_instance_capacity) {
        env->generic_func_instance_capacity *= 2;
        env->generic_func_instances = realloc(env->generic_func_instances,
            sizeof(GenericFuncInstance) * env->generic_func_instance_capacity);
    }
    GenericFuncInstance inst;
    inst.orig_name = strdup(orig_name);
    inst.mono_name = strdup(mono_name);
    inst.binding_count = binding_count;
    inst.var_names = malloc(sizeof(char*) * binding_count);
    inst.bound_types = malloc(sizeof(Type) * binding_count);
    inst.bound_type_names = malloc(sizeof(char*) * binding_count);
    for (int i = 0; i < binding_count; i++) {
        inst.var_names[i] = strdup(var_names[i]);
        inst.bound_types[i] = bound_types[i];
        inst.bound_type_names[i] = bound_type_names[i] ? strdup(bound_type_names[i]) : NULL;
    }
    env->generic_func_instances[env->generic_func_instance_count++] = inst;
}
