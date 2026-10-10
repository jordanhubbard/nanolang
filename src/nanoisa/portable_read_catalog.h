#ifndef NANOISA_PORTABLE_READ_CATALOG_H
#define NANOISA_PORTABLE_READ_CATALOG_H
#include "nvm_format.h"
#include "isa.h"
#include <string.h>

/* I recognize declarations only. Runtime bindings separately supply authority. */
static inline bool nvm_portable_read_import_exact(const NvmModule *m,uint32_t index) {
    if(!m || index>=m->import_count || !m->imports || !m->import_param_types ||
       !m->strings || !m->string_lengths)return false;
    const NvmImportEntry *imp=&m->imports[index];
    if(imp->module_name_idx>=m->string_count || imp->function_name_idx>=m->string_count ||
       imp->kind!=NVM_IMPORT_FFI || imp->param_count!=1 || imp->return_type!=TAG_STRING ||
       !m->import_param_types[index] || m->import_param_types[index][0]!=TAG_STRING ||
       m->string_lengths[imp->module_name_idx])return false;
    const char *name=m->strings[imp->function_name_idx];
    if(!name)return false;
    static const char *const names[]={"file_read","vm_file_read","nl_os_file_read"};
    for(size_t i=0;i<sizeof names/sizeof names[0];i++) {
        size_t n=strlen(names[i]);
        if(m->string_lengths[imp->function_name_idx]==n && !memcmp(name,names[i],n))return true;
    }
    return false;
}
#endif
