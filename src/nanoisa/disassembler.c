/*
 * NanoISA Disassembler
 *
 * Decodes NVM bytecode back to text assembly format.
 * Reconstructs labels from jump targets for readability.
 */

#include "disassembler.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Label Reconstruction
 *
 * Scan bytecode for jump instructions, collect their targets,
 * and assign label names (L0, L1, ...).
 * ======================================================================== */

#define MAX_DISASM_LABELS 512

typedef struct {
    uint32_t offset;
    char name[16];
} DisasmLabel;

static uint32_t collect_jump_targets(const uint8_t *code, uint32_t code_size,
                                      DisasmLabel *labels, uint32_t max_labels) {
    uint32_t label_count = 0;
    uint32_t pos = 0;

    while (pos < code_size) {
        DecodedInstruction instr;
        uint32_t consumed = isa_decode(code + pos, code_size - pos, &instr);
        if (consumed == 0) break;

        /* Check if this instruction has an I32 operand (jump offset) */
        const InstructionInfo *info = isa_get_info(instr.opcode);
        if (info) {
            for (int i = 0; i < info->operand_count; i++) {
                if (info->operands[i] == OPERAND_I32) {
                    /* Relative offset from instruction start */
                    int32_t rel = instr.operands[i].i32;
                    uint32_t target = (uint32_t)((int32_t)pos + rel);
                    if (target <= code_size) {
                        /* Check if we already have this target */
                        bool found = false;
                        for (uint32_t j = 0; j < label_count; j++) {
                            if (labels[j].offset == target) {
                                found = true;
                                break;
                            }
                        }
                        if (!found && label_count < max_labels) {
                            labels[label_count].offset = target;
                            snprintf(labels[label_count].name,
                                     sizeof(labels[label_count].name),
                                     "L%u", label_count);
                            label_count++;
                        }
                    }
                }
            }
        }

        pos += consumed;
    }

    return label_count;
}

static const char *find_label_at(const DisasmLabel *labels, uint32_t count, uint32_t offset) {
    for (uint32_t i = 0; i < count; i++) {
        if (labels[i].offset == offset) {
            return labels[i].name;
        }
    }
    return NULL;
}

/* ========================================================================
 * Operand Formatting
 * ======================================================================== */

static void format_operand(FILE *out, const DecodedInstruction *instr, int idx,
                            const NvmModule *mod, uint32_t instr_offset,
                            const DisasmLabel *labels, uint32_t label_count) {
    switch (instr->operand_types[idx]) {
        case OPERAND_U8:
            fprintf(out, " %u", instr->operands[idx].u8);
            break;
        case OPERAND_U16:
            fprintf(out, " %u", instr->operands[idx].u16);
            break;
        case OPERAND_U32:
            /* For PUSH_STR, show the actual string */
            if (instr->opcode == OP_PUSH_STR && idx == 0 && mod) {
                const char *str = nvm_get_string(mod, instr->operands[idx].u32);
                if (str) {
                    fprintf(out, " %u", instr->operands[idx].u32);
                    fprintf(out, "  ; \"%s\"", str);
                    return;
                }
            }
            /* For CALL, show function name */
            if ((instr->opcode == OP_CALL || instr->opcode == OP_CALL_EXTERN) && idx == 0 && mod) {
                uint32_t fn_idx = instr->operands[idx].u32;
                if (fn_idx < mod->function_count) {
                    const char *name = nvm_get_string(mod, mod->functions[fn_idx].name_idx);
                    if (name) {
                        fprintf(out, " %u", fn_idx);
                        fprintf(out, "  ; %s", name);
                        return;
                    }
                }
            }
            fprintf(out, " %u", instr->operands[idx].u32);
            break;
        case OPERAND_I32: {
            int32_t rel = instr->operands[idx].i32;
            uint32_t target = (uint32_t)((int32_t)instr_offset + rel);
            const char *label = find_label_at(labels, label_count, target);
            if (label) {
                fprintf(out, " %s", label);
            } else {
                fprintf(out, " %d", rel);
            }
            break;
        }
        case OPERAND_I64:
            fprintf(out, " %lld", (long long)instr->operands[idx].i64);
            break;
        case OPERAND_F64:
            fprintf(out, " %.17g", instr->operands[idx].f64);
            break;
        case OPERAND_NONE:
            break;
    }
}

/* ========================================================================
 * Function Disassembly
 * ======================================================================== */

void disasm_function(const uint8_t *code, uint32_t code_size,
                     const NvmModule *mod, FILE *out) {
    /* Collect jump targets for label reconstruction */
    DisasmLabel labels[MAX_DISASM_LABELS];
    uint32_t label_count = collect_jump_targets(code, code_size, labels, MAX_DISASM_LABELS);

    uint32_t pos = 0;
    while (pos < code_size) {
        /* Check if there's a label at this offset */
        const char *label = find_label_at(labels, label_count, pos);
        if (label) {
            fprintf(out, "%s:\n", label);
        }

        DecodedInstruction instr;
        uint32_t consumed = isa_decode(code + pos, code_size - pos, &instr);
        if (consumed == 0) {
            fprintf(out, "  ; ERROR: invalid opcode 0x%02x at offset %u\n",
                    code[pos], pos);
            pos++;
            continue;
        }

        const InstructionInfo *info = isa_get_info(instr.opcode);
        fprintf(out, "  %s", info ? info->name : "???");

        for (int i = 0; i < instr.operand_count; i++) {
            format_operand(out, &instr, i, mod, pos, labels, label_count);
        }

        fprintf(out, "\n");
        pos += consumed;
    }

    /* Check for a label at the end (loop exit target) */
    const char *end_label = find_label_at(labels, label_count, pos);
    if (end_label) {
        fprintf(out, "%s:\n", end_label);
    }
}

/* ========================================================================
 * Module Disassembly
 * ======================================================================== */

void disasm_module_to_file(const NvmModule *mod, FILE *out) {
    /* String pool */
    for (uint32_t i = 0; i < mod->string_count; i++) {
        const char *s = nvm_get_string(mod, i);
        if (s) {
            fprintf(out, ".string \"");
            /* Escape special characters */
            for (const char *p = s; *p; p++) {
                switch (*p) {
                    case '\n': fprintf(out, "\\n"); break;
                    case '\t': fprintf(out, "\\t"); break;
                    case '\\': fprintf(out, "\\\\"); break;
                    case '"':  fprintf(out, "\\\""); break;
                    default:   fputc(*p, out); break;
                }
            }
            fprintf(out, "\"\n");
        }
    }
    if (mod->string_count > 0) {
        fprintf(out, "\n");
    }

    /* Entry point */
    if (mod->header.flags & NVM_FLAG_HAS_MAIN) {
        fprintf(out, ".entry %u\n\n", mod->header.entry_point);
    }

    /* Functions */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        const char *name = nvm_get_string(mod, fn->name_idx);

        fprintf(out, ".function %s %u %u %u\n",
                name ? name : "???",
                fn->arity, fn->local_count, fn->upvalue_count);

        if (fn->code_length > 0 && fn->code_offset + fn->code_length <= mod->code_size) {
            disasm_function(mod->code + fn->code_offset, fn->code_length, mod, out);
        }

        fprintf(out, ".end\n\n");
    }
}

char *disasm_module(const NvmModule *mod) {
    /* Write to a temporary memory stream */
    char *buf = NULL;
    size_t buf_size = 0;
    FILE *stream = open_memstream(&buf, &buf_size);
    if (!stream) return NULL;

    disasm_module_to_file(mod, stream);
    fclose(stream);

    return buf;
}
