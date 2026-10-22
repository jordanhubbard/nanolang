#ifndef NANOVIRT_CHECKED_BORROW_CODEGEN_H
#define NANOVIRT_CHECKED_BORROW_CODEGEN_H
#include "codegen.h"
/* I consume only the sealed ledger owned by this synchronous Environment. */
CodegenResult codegen_compile_checked(ASTNode *,Environment *,ModuleList *,const char *);
#endif
