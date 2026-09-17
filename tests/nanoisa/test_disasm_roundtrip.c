/*
 * Canonical disassembly must be lossless.
 *
 * "Lossless" is a byte-level claim about the whole module, not just the
 * instruction stream: disassembling a module and reassembling the text must
 * reproduce the same bytecode AND the same tables. The instruction stream
 * already round-tripped; the tables did not, because the import table, the
 * linked-module references and the type-definition counts had no textual form
 * at all. A module with an import could not be reassembled -- CALL_EXTERN 0
 * referred to an import table the text never declared.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvm_format.h"
#include "isa.h"
#include "assembler.h"
#include "disassembler.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, what) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s  (%s:%d)\n", (what), __FILE__, __LINE__); } \
} while (0)

/* Assemble, disassemble canonically, reassemble, and compare everything a
 * consumer of the module can observe. */
static void round_trips(const char *label, const char *src) {
    AsmResult ar;
    NvmModule *a = asm_assemble(src, &ar);
    if (!a) { g_fail++; printf("  FAIL: %s did not assemble: %s\n", label, ar.message); return; }

    char *text = disasm_module_styled(a, DISASM_STYLE_CANONICAL);
    if (!text) { g_fail++; printf("  FAIL: %s did not disassemble\n", label); nvm_module_free(a); return; }

    AsmResult br;
    NvmModule *b = asm_assemble(text, &br);
    if (!b) {
        g_fail++;
        printf("  FAIL: %s did not reassemble: %s\n", label, br.message);
        printf("--- canonical text ---\n%s----------------------\n", text);
        free(text); nvm_module_free(a);
        return;
    }

    CHECK(a->code_size == b->code_size && memcmp(a->code, b->code, a->code_size) == 0,
          label);
    CHECK(a->function_count == b->function_count, label);
    CHECK(a->string_count == b->string_count, label);
    CHECK(a->import_count == b->import_count, label);
    CHECK(a->module_ref_count == b->module_ref_count, label);
    CHECK(a->struct_count == b->struct_count
          && a->enum_count == b->enum_count
          && a->union_count == b->union_count, label);
    CHECK(a->header.flags == b->header.flags, label);
    CHECK(a->header.entry_point == b->header.entry_point, label);

    for (uint32_t i = 0; i < a->import_count && i < b->import_count; i++) {
        const NvmImportEntry *x = &a->imports[i], *y = &b->imports[i];
        CHECK(x->module_name_idx == y->module_name_idx
              && x->function_name_idx == y->function_name_idx
              && x->param_count == y->param_count
              && x->return_type == y->return_type, "import entry survives");
        CHECK(x->param_count == 0
              || memcmp(a->import_param_types[i], b->import_param_types[i],
                        x->param_count) == 0, "import parameter tags survive");
    }
    for (uint32_t i = 0; i < a->module_ref_count && i < b->module_ref_count; i++)
        CHECK(a->module_refs[i].module_name_idx == b->module_refs[i].module_name_idx,
              "module reference survives");
    for (uint32_t i = 0; i < a->string_count && i < b->string_count; i++)
        CHECK(a->string_lengths[i] == b->string_lengths[i]
              && memcmp(a->strings[i], b->strings[i], a->string_lengths[i]) == 0,
              "string pool entry survives verbatim");

    free(text);
    nvm_module_free(a);
    nvm_module_free(b);
}

static void quoted_string_boundaries(void) {
    const char *source =
        ".string punctuation \"left; # right\" ; trailing comment\n"
        ".string escaped \"quote\\\";#slash\\\\;#\" # trailing comment\n"
        ".string empty \"\" ; empty string\n"
        ".function main 0 0 0 int 1\n PUSH_I64 0 # instruction comment\n RET\n.end\n";
    AsmResult result;
    NvmModule *module = asm_assemble(source, &result);
    CHECK(module != NULL, "quoted comment markers assemble");
    if (module) {
        CHECK(module->string_count >= 3, "all quoted constants survive");
        if (module->string_count >= 3) {
        CHECK(strcmp(module->strings[0], "left; # right") == 0, "literal comment bytes survive");
        CHECK(strcmp(module->strings[1], "quote\";#slash\\;#") == 0, "escaped quotes and slashes preserve comments");
        CHECK(module->string_lengths[2] == 0, "empty quoted constant survives");
        }
        nvm_module_free(module);
    }
    round_trips("quoted comment markers", source);
    round_trips("quoted import and module names",
        ".import \"lib;#\" \"call#;\" void # comment\n"
        ".module_ref \"other;#\" ; comment\n"
        ".function main 0 0 0 int 1\n CALL_EXTERN 0\n PUSH_I64 0\n RET\n.end\n");
    const size_t lengths[] = {4095, 4096, 16384};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) {
        size_t length = lengths[i];
        char *text = malloc(length + 128);
        CHECK(text != NULL, "allocate long string fixture");
        if (!text) continue;
        size_t prefix = (size_t)sprintf(text, ".string long ");
        text[prefix++] = '"';
        memset(text + prefix, 'x', length);
        strcpy(text + prefix + length, "\"\n.function main 0 0 0 int 1\n PUSH_I64 0\n RET\n.end\n");
        module = asm_assemble(text, &result);
        CHECK(module != NULL, "long literal assembles");
        if (module) {
            CHECK(module->string_lengths[0] == length, "long literal retains full length");
            CHECK(memcmp(module->strings[0], text + prefix, length) == 0, "long literal retains all bytes");
            nvm_module_free(module);
        }
        round_trips("long quoted literal", text);
        free(text);
    }
    const char *bad[] = {".string \"unterminated", ".string \"escaped\\\"", ".string \"dangling\\", ".string \"ok\" junk"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        module = asm_assemble(bad[i], &result);
        CHECK(module == NULL && result.error == ASM_ERR_SYNTAX, "malformed quoted directive refuses");
        nvm_module_free(module);
    }
}

static void large_symbol_and_branch_tables(void) {
    const uint32_t count = 2200;
    char *text = malloc(200000);
    CHECK(text != NULL, "allocate many-symbol fixture");
    if (!text) return;
    size_t used = 0;
    for (uint32_t i = 0; i < count; i++)
        used += (size_t)sprintf(text + used, ".string s%u \"same\"\n", i);
    used += (size_t)sprintf(text + used, ".function main 0 0 0 int 1\n");
    for (uint32_t i = 0; i < count; i++)
        used += (size_t)sprintf(text + used, " JMP label%u\nlabel%u:\n", i, i);
    sprintf(text + used, " PUSH_I64 37\n RET\n.end\n");
    AsmResult result;
    NvmModule *module = asm_assemble(text, &result);
    CHECK(module != NULL, "large symbol label and patch tables assemble");
    if (module) {
        bool all_patched = true;
        for (uint32_t i = 0; i < count; i++) {
            const uint8_t *instruction = module->code + i * 5;
            if (instruction[0] != OP_JMP || instruction[1] != 5 || instruction[2] || instruction[3] || instruction[4])
                all_patched = false;
        }
        CHECK(all_patched, "every forward jump is resolved beyond old patch limit");
        nvm_module_free(module);
    }
    round_trips("large symbol and branch tables", text);
    free(text);
    const char *duplicates[] = {
        ".string same \"one\"\n.string same \"two\"\n",
        ".function main 0 0 0 int 1\nloop:\nloop:\n PUSH_I64 0\n RET\n.end\n"
    };
    for (size_t i = 0; i < 2; i++) {
        module = asm_assemble(duplicates[i], &result);
        CHECK(module == NULL && result.error == (i == 0 ? ASM_ERR_DUPLICATE_SYMBOL : ASM_ERR_DUPLICATE_LABEL),
              "actual duplicate retains its error category");
        nvm_module_free(module);
    }
}

int main(void) {
    printf("\n[disasm_roundtrip] canonical disassembly is lossless...\n\n");
    quoted_string_boundaries();
    large_symbol_and_branch_tables();

    round_trips("simple",
        ".function main 0 1 0 int 1\n  PUSH_I64 42\n  RET\n.end\n");

    round_trips("branches and labels",
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 1\n  JMP_TRUE t\n  PUSH_I64 0\n  JMP e\n"
        "t:\n  PUSH_I64 9\ne:\n  RET\n.end\n");

    round_trips("portable ISA",
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 3\n  PUSH_I64 4\n  I64_ADD\n  RET\n.end\n");

    /* An embedded zero, a carriage return and a high byte: the pool carries
     * explicit lengths, so all three must come back verbatim rather than
     * truncated at the zero. */
    round_trips("binary string constants",
        ".string \"a\\x00b\\r\\xff\"\n"
        ".function main 0 1 0 string 1\n  PUSH_STR 0\n  RET\n.end\n");

    round_trips("two functions and a direct call",
        ".function helper 1 1 0 int 1\n  LOAD_LOCAL 0\n  RET\n.end\n"
        ".function main 0 1 0 int 1\n  PUSH_I64 1\n  CALL 0\n  RET\n.end\n");

    round_trips("indirect call shape",
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 1\n  FUNCREF 0\n  CALL_INDIRECT 1 1\n  RET\n.end\n");

    /* The cases the textual form could not express at all. */
    round_trips("an import table",
        ".string \"libm\"\n.string \"sqrt\"\n"
        ".import \"libm\" \"sqrt\" float float\n"
        ".function main 0 1 0 float 1\n"
        "  PUSH_F64 4.0\n  CALL_EXTERN 0\n  RET\n.end\n");

    round_trips("a nullary void import",
        ".string \"libc\"\n.string \"abort\"\n"
        ".import \"libc\" \"abort\" void\n"
        ".function main 0 1 0 int 1\n"
        "  CALL_EXTERN 0\n  PUSH_I64 0\n  RET\n.end\n");

    round_trips("linked module references",
        ".string \"other\"\n"
        ".module_ref \"other\"\n"
        ".function main 0 1 0 int 1\n"
        "  CALL_MODULE 0 0 0 1\n  RET\n.end\n");

    round_trips("type definition counts",
        ".types 3 2 1\n"
        ".function main 0 1 0 int 1\n  PUSH_I64 1\n  RET\n.end\n");

    round_trips("everything at once",
        ".string \"libm\"\n.string \"sqrt\"\n.string \"other\"\n"
        ".types 2 1 1\n"
        ".import \"libm\" \"sqrt\" float float\n"
        ".module_ref \"other\"\n"
        ".flag needs_extern\n"
        ".function main 0 1 0 float 1\n"
        "  PUSH_F64 9.0\n  CALL_EXTERN 0\n  RET\n.end\n");

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
