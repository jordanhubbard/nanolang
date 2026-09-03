/*
 * NanoISA Disassembler
 *
 * Decodes NVM bytecode back to text assembly format.
 * Reconstructs labels from jump targets for readability.
 */

#define _POSIX_C_SOURCE 200809L  /* For open_memstream() */

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

/* Return the operand index that carries a relative branch target for this
 * opcode, or -1 if the opcode has no branch operand. Only these operands are
 * branch offsets that resolve to labels; other I32 operands are plain
 * immediates and must be printed numerically. */
static int branch_operand_index(NanoOpcode opcode) {
    switch (opcode) {
        case OP_JMP:
        case OP_JMP_TRUE:
        case OP_JMP_FALSE:
            return 0;
        case OP_MATCH_TAG:
            return 1;
        default:
            return -1;
    }
}

static uint32_t collect_jump_targets(const uint8_t *code, uint32_t code_size,
                                      DisasmLabel *labels, uint32_t max_labels) {
    uint32_t label_count = 0;
    uint32_t pos = 0;

    while (pos < code_size) {
        DecodedInstruction instr;
        uint32_t consumed = isa_decode(code + pos, code_size - pos, &instr);
        if (consumed == 0) break;

        /* Only the branch operand of a control-flow opcode is a jump target;
         * other I32 operands (e.g. immediates) must not create labels. */
        int branch_idx = branch_operand_index(instr.opcode);
        if (branch_idx >= 0 && branch_idx < instr.operand_count &&
            instr.operand_types[branch_idx] == OPERAND_I32) {
            /* Relative offset from instruction start */
            int32_t rel = instr.operands[branch_idx].i32;
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

static bool is_control_flow_opcode(NanoOpcode opcode) {
    return opcode == OP_JMP ||
           opcode == OP_JMP_FALSE ||
           opcode == OP_JMP_TRUE ||
           opcode == OP_MATCH_TAG ||
           opcode == OP_CALL ||
           opcode == OP_TAIL_CALL ||
           opcode == OP_CALL_EXTERN ||
           opcode == OP_RET ||
           opcode == OP_HALT;
}

static const char *control_flow_note(NanoOpcode opcode) {
    switch (opcode) {
        case OP_JMP: return "jump";
        case OP_JMP_FALSE: return "branch-if-false";
        case OP_JMP_TRUE: return "branch-if-true";
        case OP_MATCH_TAG: return "match-tag-branch";
        case OP_CALL: return "call";
        case OP_TAIL_CALL: return "tail-call";
        case OP_CALL_EXTERN: return "extern-call";
        case OP_RET: return "return";
        case OP_HALT: return "halt";
        default: return NULL;
    }
}

/* ========================================================================
 * Operand Formatting
 * ======================================================================== */

static void format_operand(FILE *out, const DecodedInstruction *instr, int idx,
                            const NvmModule *mod, uint32_t instr_offset,
                            const DisasmLabel *labels, uint32_t label_count,
                            DisasmStyle style) {
    switch (instr->operand_types[idx]) {
        case OPERAND_U8:
            fprintf(out, " %u", instr->operands[idx].u8);
            break;
        case OPERAND_U16:
            fprintf(out, " %u", instr->operands[idx].u16);
            break;
        case OPERAND_U32:
            /* For PUSH_STR, show the actual string */
            if (style == DISASM_STYLE_DETAILED &&
                instr->opcode == OP_PUSH_STR && idx == 0 && mod) {
                const char *str = nvm_get_string(mod, instr->operands[idx].u32);
                if (str) {
                    fprintf(out, " %u", instr->operands[idx].u32);
                    fprintf(out, "  ; \"%s\"", str);
                    return;
                }
            }
            /* For CALL, show function name */
            if (style == DISASM_STYLE_DETAILED &&
                (instr->opcode == OP_CALL || instr->opcode == OP_TAIL_CALL) &&
                idx == 0 && mod) {
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
            /* For CALL_EXTERN, the operand is an import table index (not a
             * function table index): annotate with the imported module and
             * function name so external calls read correctly. */
            if (style == DISASM_STYLE_DETAILED &&
                instr->opcode == OP_CALL_EXTERN && idx == 0 && mod) {
                uint32_t imp_idx = instr->operands[idx].u32;
                if (imp_idx < mod->import_count) {
                    const NvmImportEntry *imp = &mod->imports[imp_idx];
                    const char *mod_name = nvm_get_string(mod, imp->module_name_idx);
                    const char *fn_name = nvm_get_string(mod, imp->function_name_idx);
                    fprintf(out, " %u", imp_idx);
                    if (mod_name && fn_name) {
                        fprintf(out, "  ; import %s.%s", mod_name, fn_name);
                    } else if (fn_name) {
                        fprintf(out, "  ; import %s", fn_name);
                    } else {
                        fprintf(out, "  ; import");
                    }
                    return;
                }
            }
            fprintf(out, " %u", instr->operands[idx].u32);
            break;
        case OPERAND_I32: {
            int32_t rel = instr->operands[idx].i32;
            /* Only the branch operand resolves to a label; other I32 operands
             * are plain signed immediates and print numerically. */
            if (branch_operand_index(instr->opcode) == idx) {
                uint32_t target = (uint32_t)((int32_t)instr_offset + rel);
                const char *label = find_label_at(labels, label_count, target);
                if (label) {
                    fprintf(out, " %s", label);
                    break;
                }
            }
            fprintf(out, " %d", rel);
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

void disasm_function_styled(const uint8_t *code, uint32_t code_size,
                            const NvmModule *mod, FILE *out,
                            DisasmStyle style) {
    /* Collect jump targets for label reconstruction */
    DisasmLabel labels[MAX_DISASM_LABELS];
    uint32_t label_count = collect_jump_targets(code, code_size, labels, MAX_DISASM_LABELS);

    uint32_t function_base_offset = 0;
    if (mod && mod->code && code >= mod->code && code <= mod->code + mod->code_size) {
        function_base_offset = (uint32_t)(code - mod->code);
    }

    const char *src_file = "<unknown>";
    if (mod) {
        if (mod->source_file_idx > 0) {
            const char *source_string = nvm_get_string(mod, mod->source_file_idx);
            if (source_string && source_string[0]) src_file = source_string;
        } else {
            const char *fallback_string = nvm_get_string(mod, 0);
            if (fallback_string && fallback_string[0]) src_file = fallback_string;
        }
    }

    uint32_t current_line = 0;

    uint32_t pos = 0;
    while (pos < code_size) {
        /* Check if there's a label at this offset */
        const char *label = find_label_at(labels, label_count, pos);
        if (label) {
            if (style == DISASM_STYLE_CANONICAL) {
                fprintf(out, "%s:\n", label);
            } else {
                fprintf(out, "%s:  ; <== jump target\n", label);
            }
        }

        DecodedInstruction instr;
        uint32_t consumed = isa_decode(code + pos, code_size - pos, &instr);
        if (consumed == 0) {
            fprintf(out, "  ; ERROR: invalid opcode 0x%02x at offset %u (abs %u)\n",
                    code[pos], pos, function_base_offset + pos);
            pos++;
            continue;
        }

        const InstructionInfo *info = isa_get_info(instr.opcode);
        if (style == DISASM_STYLE_CANONICAL) {
            fprintf(out, "  %s", info ? info->name : "???");
        } else {
            fprintf(out, "  [%04u|%04u] %s", pos, function_base_offset + pos,
                    info ? info->name : "???");
        }

        for (int i = 0; i < instr.operand_count; i++) {
            format_operand(out, &instr, i, mod, pos, labels, label_count, style);
        }

        if (style == DISASM_STYLE_DETAILED) {
            if (instr.opcode == OP_DEBUG_LINE && instr.operand_count > 0) {
                current_line = instr.operands[0].u32;
                if (current_line > 0) {
                    fprintf(out, "  ; source %s:%u", src_file, current_line);
                }
            } else if (current_line > 0) {
                fprintf(out, "  ; @ %s:%u", src_file, current_line);
            }

            if (is_control_flow_opcode(instr.opcode)) {
                const char *note = control_flow_note(instr.opcode);
                if (note) fprintf(out, "  ; cfg:%s", note);
            }
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

void disasm_function(const uint8_t *code, uint32_t code_size,
                     const NvmModule *mod, FILE *out) {
    disasm_function_styled(code, code_size, mod, out, DISASM_STYLE_DETAILED);
}

/* ========================================================================
 * Module Disassembly
 * ======================================================================== */

void disasm_module_to_file_styled(const NvmModule *mod, FILE *out,
                                  DisasmStyle style) {
    /* String pool */
    for (uint32_t i = 0; i < mod->string_count; i++) {
        const char *s = nvm_get_string(mod, i);
        if (s) {
            /* Emit exactly the stored bytes: strings may contain embedded NUL
             * and other non-printable bytes, so iterate by length rather than
             * stopping at the first NUL. Non-printable bytes are emitted as
             * \xHH escapes so binary strings round-trip losslessly. */
            uint32_t len = nvm_get_string_len(mod, i);
            fprintf(out, ".string \"");
            for (uint32_t j = 0; j < len; j++) {
                unsigned char c = (unsigned char)s[j];
                switch (c) {
                    case '\n': fprintf(out, "\\n"); break;
                    case '\r': fprintf(out, "\\r"); break;
                    case '\t': fprintf(out, "\\t"); break;
                    case '\\': fprintf(out, "\\\\"); break;
                    case '"':  fprintf(out, "\\\""); break;
                    default:
                        if (c < 0x20 || c >= 0x7f) {
                            fprintf(out, "\\x%02x", c);
                        } else {
                            fputc((int)c, out);
                        }
                        break;
                }
            }
            fprintf(out, "\"\n");
        }
    }
    if (mod->string_count > 0) {
        fprintf(out, "\n");
    }

    if (style == DISASM_STYLE_CANONICAL) {
        if (mod->header.flags & NVM_FLAG_NEEDS_EXTERN) {
            fprintf(out, ".flag needs_extern\n");
        }
        if (mod->header.flags & NVM_FLAG_DEBUG_INFO) {
            fprintf(out, ".flag debug_info\n");
        }
        if ((mod->header.flags & (NVM_FLAG_NEEDS_EXTERN | NVM_FLAG_DEBUG_INFO)) != 0) {
            fprintf(out, "\n");
        }
    }

    /* Entry point */
    if (mod->header.flags & NVM_FLAG_HAS_MAIN) {
        fprintf(out, ".entry %u\n\n", mod->header.entry_point);
    }

    /* Functions */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        const char *name = nvm_get_string(mod, fn->name_idx);

        fprintf(out, ".function %s %u %u %u %s %u\n",
                name ? name : "???",
                fn->arity, fn->local_count, fn->upvalue_count,
                isa_tag_name(fn->result_tag), fn->result_count);

        if (fn->code_length > 0 && fn->code_offset <= mod->code_size &&
            fn->code_length <= mod->code_size - fn->code_offset) {
            disasm_function_styled(mod->code + fn->code_offset, fn->code_length,
                                   mod, out, style);
        }

        fprintf(out, ".end\n\n");
    }
}

void disasm_module_to_file(const NvmModule *mod, FILE *out) {
    disasm_module_to_file_styled(mod, out, DISASM_STYLE_DETAILED);
}

char *disasm_module_styled(const NvmModule *mod, DisasmStyle style) {
    char *buf = NULL;
    size_t buf_size = 0;
    FILE *stream = open_memstream(&buf, &buf_size);
    if (!stream) return NULL;

    disasm_module_to_file_styled(mod, stream, style);
    fclose(stream);

    return buf;
}

char *disasm_module(const NvmModule *mod) {
    return disasm_module_styled(mod, DISASM_STYLE_DETAILED);
}
