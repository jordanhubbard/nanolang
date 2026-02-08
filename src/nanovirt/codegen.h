/*
 * codegen.h - NanoISA bytecode generation from nanolang AST
 *
 * Compiles a typechecked AST into an NvmModule containing bytecode.
 * Reuses the existing frontend (lexer/parser/typechecker) and targets
 * the NanoISA instruction set defined in nanoisa/isa.h.
 */
#ifndef NANOVIRT_CODEGEN_H
#define NANOVIRT_CODEGEN_H

#include "nanolang.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"

#include <stdbool.h>
#include <stdint.h>

/* Result of compilation */
typedef struct {
    NvmModule *module;      /* NULL on error */
    bool ok;
    int error_line;
    char error_msg[256];
} CodegenResult;

/*
 * Compile a typechecked AST program into an NvmModule.
 *
 * @param program  AST root (must be AST_PROGRAM)
 * @param env      Post-typechecked environment
 * @return         CodegenResult with module or error info
 */
CodegenResult codegen_compile(ASTNode *program, Environment *env);

#endif /* NANOVIRT_CODEGEN_H */
