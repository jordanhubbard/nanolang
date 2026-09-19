#ifndef NL_NSI_FILE_CATALOG_H
#define NL_NSI_FILE_CATALOG_H
#include "nsi_file_plan.h"
/* Private, immutable process-lifetime views of catalog1. No owned plan or
 * execution authority is created. I use the same facts as document checking. */
const char *nl_file_catalog_interface(void);
const NlFilePlanMethod *nl_file_catalog_method(size_t);
const NlFilePlanType *nl_file_catalog_type(size_t);
#endif
