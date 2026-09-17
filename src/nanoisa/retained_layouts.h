#ifndef NANOISA_RETAINED_LAYOUTS_H
#define NANOISA_RETAINED_LAYOUTS_H
#include "nvm_format.h"
#include "nvm_v2_sections.h"

/* I distinguish actual layout facts from legacy count-only placeholders. */
bool nvm_layouts_have_facts(const NvmV2Layouts *layouts);
/* I validate owned canonical layout bytes against execution-module counts
 * and string indices. Absence is compatible with legacy untyped modules. */
bool nvm_retained_layouts_valid(const NvmModule *module);
/* I copy the table without changing its layout indices or per-kind order.
 * Counts and strings must already be installed. Failure preserves old data. */
NvmV2Result nvm_retain_layouts(NvmModule *module, const NvmV2Layouts *layouts);
#endif
