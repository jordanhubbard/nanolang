#ifndef NANO_GENERIC_BINDINGS_H
#define NANO_GENERIC_BINDINGS_H
#include "nanolang.h"

/* I share structural inference and owned substitution between checking and emission. */
typedef struct {
    const char *names[26];
    TypeInfo *types[26];
    int count;
} GenericBindings;

static inline bool generic_variable(Environment *env, const TypeInfo *info) {
    const char *name = info ? info->generic_name : NULL;
    return info && info->base_type == TYPE_STRUCT && name &&
        name[0] >= 'A' && name[0] <= 'Z' && !name[1] &&
        !env_get_struct(env, name) && !env_get_enum(env, name) &&
        !env_get_union(env, name) && !env_get_opaque_type(env, name);
}
static inline const TypeInfo *generic_view(const TypeInfo *info, Type type,
        const char *name, FunctionSignature *signature, TypeInfo *fallback) {
    if (info) return info;
    *fallback = (TypeInfo){.base_type=type, .generic_name=(char *)name, .fn_sig=signature};
    return fallback;
}
static inline const TypeInfo *generic_parameter(const Parameter *param, TypeInfo *fallback) {
    return generic_view(param->type_info, param->type, param->struct_type_name, param->fn_sig, fallback);
}
static inline const TypeInfo *generic_signature_param(const FunctionSignature *sig, int p, TypeInfo *fallback) {
    return generic_view(sig->param_type_info ? sig->param_type_info[p] : NULL,
        sig->param_types[p], sig->param_struct_names ? sig->param_struct_names[p] : NULL, NULL, fallback);
}
static inline const TypeInfo *generic_signature_result(const FunctionSignature *sig, TypeInfo *fallback) {
    return generic_view(sig->return_type_info, sig->return_type, sig->return_struct_name, sig->return_fn_sig, fallback);
}
static inline bool generic_contains(Environment *env, const TypeInfo *info, unsigned depth) {
    if (!info || depth > 128) return false;
    if (generic_variable(env, info)) return true;
    if (generic_contains(env, info->element_type, depth+1)) return true;
    for (int p=0; p<info->type_param_count; ++p)
        if (generic_contains(env, info->type_params[p], depth+1)) return true;
    if (info->fn_sig) {
        TypeInfo view;
        for (int p=0; p<info->fn_sig->param_count; ++p)
            if (generic_contains(env, generic_signature_param(info->fn_sig,p,&view), depth+1)) return true;
        if (generic_contains(env, generic_signature_result(info->fn_sig,&view), depth+1)) return true;
    }
    return false;
}
static inline bool generic_function(Environment *env, const Function *function) {
    if (!function || !function->params) return false;
    for (int p=0; p<function->param_count; ++p) {
        TypeInfo view;
        if (generic_contains(env, generic_parameter(&function->params[p], &view),0)) return true;
    }
    return false;
}
static inline void generic_bindings_free(GenericBindings *bindings) {
    for (int i=0; i<bindings->count; ++i) free_payload_type_info(bindings->types[i]);
    *bindings=(GenericBindings){0};
}
static inline bool generic_bind(Environment *env, const TypeInfo *formal,
        const TypeInfo *actual, GenericBindings *bindings, unsigned depth) {
    if (!formal || !actual || depth>128 || actual->base_type==TYPE_UNKNOWN) return false;
    if (generic_variable(env,formal)) {
        for (int i=0; i<bindings->count; ++i)
            if (!strcmp(bindings->names[i],formal->generic_name))
                return type_infos_equal(bindings->types[i],actual);
        if (bindings->count==26) return false;
        int i=bindings->count++;
        bindings->names[i]=formal->generic_name;
        bindings->types[i]=copy_payload_type_info(actual);
        return bindings->types[i]!=NULL;
    }
    if (!generic_contains(env,formal,depth)) return type_infos_equal(formal,actual);
    if (formal->base_type!=actual->base_type || formal->type_param_count!=actual->type_param_count) return false;
    if (formal->element_type || actual->element_type) {
        if (!generic_bind(env,formal->element_type,actual->element_type,bindings,depth+1)) return false;
    }
    if (formal->type_param_count) {
        if (!formal->generic_name || !actual->generic_name || strcmp(formal->generic_name,actual->generic_name)) return false;
        for (int p=0;p<formal->type_param_count;++p)
            if (!generic_bind(env,formal->type_params[p],actual->type_params[p],bindings,depth+1)) return false;
    }
    if (formal->fn_sig || actual->fn_sig) {
        const FunctionSignature *f=formal->fn_sig,*a=actual->fn_sig;
        if (!f || !a || f->param_count!=a->param_count) return false;
        TypeInfo fv,av;
        for (int p=0;p<f->param_count;++p)
            if (!generic_bind(env,generic_signature_param(f,p,&fv),generic_signature_param(a,p,&av),bindings,depth+1)) return false;
        if (!generic_bind(env,generic_signature_result(f,&fv),generic_signature_result(a,&av),bindings,depth+1)) return false;
    }
    return true;
}
static inline TypeInfo *generic_substitute(Environment *,const TypeInfo *,const GenericBindings *,unsigned);
static inline bool generic_substitute_signature(Environment *env, FunctionSignature *sig,
        const GenericBindings *bindings, unsigned depth) {
    if (sig->param_count && !sig->param_type_info) sig->param_type_info=calloc((size_t)sig->param_count,sizeof(TypeInfo *));
    if (sig->param_count && !sig->param_struct_names) sig->param_struct_names=calloc((size_t)sig->param_count,sizeof(char *));
    if (sig->param_count && (!sig->param_type_info || !sig->param_struct_names)) return false;
    TypeInfo view;
    for (int p=0;p<sig->param_count;++p) {
        TypeInfo *info=generic_substitute(env,generic_signature_param(sig,p,&view),bindings,depth+1);
        if (!info) return false;
        free_payload_type_info(sig->param_type_info[p]);sig->param_type_info[p]=info;
        sig->param_types[p]=info->base_type;
        free(sig->param_struct_names[p]);sig->param_struct_names[p]=info->generic_name?strdup(info->generic_name):NULL;
    }
    TypeInfo *info=generic_substitute(env,generic_signature_result(sig,&view),bindings,depth+1);
    if (!info) return false;
    free_payload_type_info(sig->return_type_info);sig->return_type_info=info;
    sig->return_type=info->base_type;
    free(sig->return_struct_name);sig->return_struct_name=info->generic_name?strdup(info->generic_name):NULL;
    free_function_signature(sig->return_fn_sig);sig->return_fn_sig=copy_function_signature(info->fn_sig);
    return true;
}
static inline TypeInfo *generic_substitute(Environment *env, const TypeInfo *formal,
        const GenericBindings *bindings, unsigned depth) {
    if (!formal || depth>128) return NULL;
    if (generic_variable(env,formal)) {
        for (int i=0;i<bindings->count;++i)
            if (!strcmp(bindings->names[i],formal->generic_name)) return copy_payload_type_info(bindings->types[i]);
        return NULL;
    }
    TypeInfo *copy=copy_payload_type_info(formal);
    if (copy->element_type) {
        free_payload_type_info(copy->element_type);
        copy->element_type=generic_substitute(env,formal->element_type,bindings,depth+1);
        if (!copy->element_type) { free_payload_type_info(copy); return NULL; }
    }
    for (int p=0;p<copy->type_param_count;++p) {
        free_payload_type_info(copy->type_params[p]);
        copy->type_params[p]=generic_substitute(env,formal->type_params[p],bindings,depth+1);
        if (!copy->type_params[p]) { free_payload_type_info(copy); return NULL; }
    }
    if (copy->fn_sig && !generic_substitute_signature(env,copy->fn_sig,bindings,depth+1)) { free_payload_type_info(copy); return NULL; }
    return copy;
}
#endif
