#ifndef NANOISA_PORTABLE_READ_CATALOG_H
#define NANOISA_PORTABLE_READ_CATALOG_H
#include "nvm_format.h"
#include "isa.h"
#include <string.h>

/* I recognize declarations only. Runtime bindings separately supply authority. */
/* 0 is unsupported, 1 is text, 2 is packed bytes. */
static inline unsigned nvm_portable_file_read_import(const NvmModule *m,uint32_t index) {
    if(!m || index>=m->import_count || !m->imports || !m->import_param_types ||
       !m->strings || !m->string_lengths)return false;
    const NvmImportEntry *imp=&m->imports[index];
    if(imp->module_name_idx>=m->string_count || imp->function_name_idx>=m->string_count ||
       imp->kind!=NVM_IMPORT_FFI || imp->param_count!=1 || (imp->return_type!=TAG_STRING && imp->return_type!=TAG_ARRAY) ||
       !m->import_param_types[index] || m->import_param_types[index][0]!=TAG_STRING ||
       m->string_lengths[imp->module_name_idx])return false;
    const char *name=m->strings[imp->function_name_idx];
    if(!name)return false;
    static const char *const names[]={"file_read","vm_file_read","nl_os_file_read"};
    for(size_t i=0;i<sizeof names/sizeof names[0];i++) {
        size_t n=strlen(names[i]);
        if(imp->return_type==TAG_STRING && m->string_lengths[imp->function_name_idx]==n && !memcmp(name,names[i],n))return 1;
        if(imp->return_type==TAG_ARRAY && m->string_lengths[imp->function_name_idx]==n+6 &&
           !memcmp(name,names[i],n) && !memcmp(name+n,"_bytes",6))return 2;
    }
    return false;
}
static inline bool nvm_portable_read_import_exact(const NvmModule *m,uint32_t index) {
    return nvm_portable_file_read_import(m,index)==1;
}
#endif
