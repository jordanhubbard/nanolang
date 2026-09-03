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

int main(void) {
    printf("\n[disasm_roundtrip] canonical disassembly is lossless...\n\n");

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
