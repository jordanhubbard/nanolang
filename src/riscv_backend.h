/*
 * riscv_backend.h — nanolang native RISC-V assembly backend
 *
 * Emits bare-metal RISC-V RV64IMA assembly (.s) from the nanolang AST.
 * Enables nano programs to run as agentOS PD code without WASM interpreter
 * or LLVM toolchain on seL4 RISC-V targets.
 *
 * Register allocation: linear scan over a fixed set of callee-saved regs.
 * Calling convention: RISC-V standard (a0-a7 args, a0 return, ra saved).
 *
 * Usage:
 *   nanoc --target riscv input.nano [-o output.s]
 *   riscv64-unknown-elf-gcc -nostdlib output.s -o output.elf
 */

#pragma once
#ifndef RISCV_BACKEND_H
#define RISCV_BACKEND_H

#include "nanolang.h"
#include <stdio.h>
#include <stdbool.h>

int riscv_backend_emit(ASTNode *root, const char *output_path,
                       const char *source_file, bool verbose);

int riscv_backend_emit_fp(ASTNode *root, FILE *out,
                          const char *source_file, bool verbose);

#endif /* RISCV_BACKEND_H */
