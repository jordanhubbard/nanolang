#ifndef NL_SERVICE_DRIVER_H
#define NL_SERVICE_DRIVER_H
#include "nanolang.h"
#include "runtime/service_product.h"
int nl_service_compile(ASTNode *, Environment *, bool include_imports,
                       const NlServiceProductOptions *);
char *nl_service_product_root(const char *executable);
#endif
