#define _POSIX_C_SOURCE 200809L

#include "forth_see.h"

#include "../nanoisa/nanoisa.h"
#include "../../src/nanoisa/disassembler.h"
#include "../../src/nanoisa/isa.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FORTH_SEE_OUTPUT_SIZE (1U << 16)
#define FORTH_WORD_LOCAL_INDEX 6

typedef struct {
    uint32_t start;
    uint32_t end;
} CodeBlock;

static char output[FORTH_SEE_OUTPUT_SIZE];
static size_t output_length;

static void append_output(const char *format, ...) {
    if (output_length >= sizeof(output) - 1) {
        return;
    }

    va_list args;
    va_start(args, format);
    int written = vsnprintf(output + output_length,
                            sizeof(output) - output_length,
                            format, args);
    va_end(args);
    if (written < 0) {
        return;
    }

    size_t available = sizeof(output) - output_length;
    if ((size_t)written >= available) {
        output_length = sizeof(output) - 1;
    } else {
        output_length += (size_t)written;
    }
}

static uint32_t find_function(const NvmModule *mod, const char *name) {
    for (uint32_t i = 0; i < mod->function_count; i++) {
        const char *candidate = nvm_get_string(mod,
                                                mod->functions[i].name_idx);
        if (candidate && strcmp(candidate, name) == 0) {
            return i;
        }
    }
    return UINT32_MAX;
}

static uint32_t find_string(const NvmModule *mod, const char *value) {
    for (uint32_t i = 0; i < mod->string_count; i++) {
        const char *candidate = nvm_get_string(mod, i);
        if (candidate && strcmp(candidate, value) == 0) {
            return i;
        }
    }
    return UINT32_MAX;
}

static bool decode_at(const uint8_t *code, uint32_t size, uint32_t offset,
                      DecodedInstruction *instruction,
                      uint32_t *next_offset) {
    if (offset >= size) {
        return false;
    }
    uint32_t consumed = isa_decode(code + offset, size - offset,
                                   instruction);
    if (consumed == 0) {
        return false;
    }
    *next_offset = offset + consumed;
    return true;
}

static CodeBlock find_word_block(const NvmModule *mod,
                                 const NvmFunctionEntry *function,
                                 uint32_t word_string_index) {
    const CodeBlock none = {0, 0};
    if (function->code_offset > mod->code_size
            || function->code_length > mod->code_size - function->code_offset) {
        return none;
    }

    const uint8_t *code = mod->code + function->code_offset;
    uint32_t size = function->code_length;
    uint32_t position = 0;
    while (position < size) {
        DecodedInstruction load_word;
        uint32_t after_load = 0;
        if (!decode_at(code, size, position, &load_word, &after_load)) {
            return none;
        }

        DecodedInstruction push_name;
        DecodedInstruction compare;
        DecodedInstruction skip;
        uint32_t after_push = 0;
        uint32_t after_compare = 0;
        uint32_t after_skip = 0;
        bool matches = load_word.opcode == OP_LOAD_LOCAL
            && load_word.operand_count == 1
            && load_word.operands[0].u16 == FORTH_WORD_LOCAL_INDEX
            && decode_at(code, size, after_load, &push_name, &after_push)
            && push_name.opcode == OP_PUSH_STR
            && push_name.operand_count == 1
            && push_name.operands[0].u32 == word_string_index
            && decode_at(code, size, after_push, &compare, &after_compare)
            && compare.opcode == OP_EQ
            && decode_at(code, size, after_compare, &skip, &after_skip)
            && skip.opcode == OP_JMP_FALSE
            && skip.operand_count == 1;

        if (matches) {
            int64_t target = (int64_t)after_compare + skip.operands[0].i32;
            if (target > after_skip && target <= size) {
                CodeBlock block = {
                    .start = function->code_offset + after_skip,
                    .end = function->code_offset + (uint32_t)target
                };
                return block;
            }
        }
        position = after_load;
    }
    return none;
}

static char *disassemble_block(const NvmModule *mod, CodeBlock block) {
    char *text = NULL;
    size_t text_size = 0;
    FILE *stream = open_memstream(&text, &text_size);
    if (!stream) {
        return NULL;
    }
    disasm_function(mod->code + block.start, block.end - block.start,
                    mod, stream);
    if (fclose(stream) != 0) {
        free(text);
        return NULL;
    }
    return text;
}

const char *nl_forth_see(const char *word_name, const char *nvm_path) {
    output[0] = '\0';
    output_length = 0;

    NanoisaErr load_error;
    NvmModule *mod = nanoisa_load_file(nvm_path, &load_error);
    if (!mod) {
        append_output("SEE: cannot load %s: %s\n",
                      nvm_path, load_error.message);
        return output;
    }

    uint32_t function_index = find_function(mod, "exec_builtin");
    if (function_index == UINT32_MAX) {
        append_output("SEE: exec_builtin not found in %s\n", nvm_path);
        nvm_module_free(mod);
        return output;
    }

    uint32_t word_string_index = find_string(mod, word_name);
    if (word_string_index == UINT32_MAX) {
        append_output("; %s: not a built-in word\n", word_name);
        nvm_module_free(mod);
        return output;
    }

    CodeBlock block = find_word_block(mod, &mod->functions[function_index],
                                      word_string_index);
    if (block.start == 0 && block.end == 0) {
        append_output(
            "; %s: control word — implemented in exec_tokens "
            "(not exec_builtin)\n",
            word_name);
        nvm_module_free(mod);
        return output;
    }

    append_output("; ISA implementation of Forth word: %s\n", word_name);
    append_output("; NanoISA block: bytes 0x%04X–0x%04X (%u bytes)\n;\n",
                  block.start, block.end, block.end - block.start);

    char *disassembly = disassemble_block(mod, block);
    if (disassembly) {
        append_output("%s", disassembly);
        free(disassembly);
    }

    nvm_module_free(mod);
    return output;
}
