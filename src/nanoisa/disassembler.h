/*
 * NanoISA Disassembler
 *
 * Converts NVM binary modules back to human-readable text assembly.
 */

#ifndef NANOISA_DISASSEMBLER_H
#define NANOISA_DISASSEMBLER_H

#include "nvm_format.h"
#include <stdio.h>

/* Disassemble a module to a dynamically allocated string.
 * Caller must free the returned string.
 * Returns NULL on error. */
char *disasm_module(const NvmModule *mod);

/* Disassemble a module to a file stream. */
void disasm_module_to_file(const NvmModule *mod, FILE *out);

/* Disassemble a single function's bytecode.
 * code: bytecode bytes for this function
 * code_size: length in bytes
 * mod: module (for string pool lookups)
 * out: output file stream */
void disasm_function(const uint8_t *code, uint32_t code_size,
                     const NvmModule *mod, FILE *out);

#endif /* NANOISA_DISASSEMBLER_H */
