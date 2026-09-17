#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "retained_layouts.h"
#include "reference_places.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "isa.h"

static unsigned checks;
#define CHECK(c) do { checks++; assert(c); } while (0)

static uint8_t *serialize(const NvmModule *module, size_t *size) {
    NvmV2Module wire;
    CHECK(nvm_v2_from_nvm_module(module, &wire) == NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&wire, NULL, 0, size) == NVM_V2_OK);
    uint8_t *bytes = malloc(*size);
    CHECK(bytes != NULL);
    CHECK(nvm_v2_module_serialize(&wire, bytes, *size, NULL) == NVM_V2_OK);
    nvm_v2_module_free(&wire);
    return bytes;
}

int main(int argc, char **argv) {
    AsmResult result;
    NvmModule *module = asm_assemble(
        ".types 3 0 0\n.entry 0\n.function main 0 0 0 int 1\n"
        "PUSH_I64 42\nAGG_PACK 0 0 0 1\n"
        "PUSH_I64 7\nAGG_PACK 0 1 0 1\nAGG_PACK 0 2 0 2\n"
        "AGG_GET 0\nAGG_GET 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n", &result);
    CHECK(module != NULL);
    uint32_t handle = nvm_add_string(module, "Handle", 6);
    uint32_t other = nvm_add_string(module, "Other", 5);
    uint32_t pair = nvm_add_string(module, "Pair", 4);
    uint32_t fd = nvm_add_string(module, "fd", 2);
    uint32_t left = nvm_add_string(module, "left", 4);
    uint32_t right = nvm_add_string(module, "right", 5);
    NvmV2LayoutField scalar = {TAG_INT, NVM_V2_NO_INDEX, fd};
    NvmV2LayoutField children[] = {{TAG_STRUCT, 0, left}, {TAG_STRUCT, 1, right}};
    NvmV2Layout items[] = {
        {NVM_V2_LAYOUT_STRUCT, 1, handle, &scalar},
        {NVM_V2_LAYOUT_STRUCT, 1, other, &scalar},
        {NVM_V2_LAYOUT_STRUCT, 2, pair, children}
    };
    NvmV2Layouts layouts = {items, 3};
    CHECK(nvm_retain_layouts(module, &layouts) == NVM_V2_OK);
    CHECK(nvm_retained_layouts_valid(module));
    CHECK(nvm_verify(module).ok);
    uint32_t legacy_size = 77;
    CHECK(nvm_serialize(module, &legacy_size) == NULL);
    CHECK(legacy_size == 0);
    size_t size;
    uint8_t *bytes = serialize(module, &size);
    NvmV2Header header;
    CHECK(nvm_v2_read_header(bytes, size, &header) == NVM_V2_OK);
    CHECK(header.feature_bits & NVM_V2_FEATURE_RETAINED_LAYOUTS);
    if (argc == 2) {
        FILE *out = fopen(argv[1], "wb"); CHECK(out != NULL);
        CHECK(fwrite(bytes, 1, size, out) == size); CHECK(fclose(out) == 0);
    }

    /* Invalid replacement preserves the installed table. */
    uint8_t *previous = module->layout_data;
    items[2].name_idx = module->string_count;
    CHECK(nvm_retain_layouts(module, &layouts) != NVM_V2_OK);
    CHECK(module->layout_data == previous);
    items[2].name_idx = pair;
    children[0].nested_idx = 2;
    CHECK(nvm_retain_layouts(module, &layouts) != NVM_V2_OK);
    CHECK(module->layout_data == previous);
    children[0].nested_idx = 0;
    module->struct_count = 2;
    CHECK(!nvm_retained_layouts_valid(module));
    CHECK(!nvm_verify(module).ok);
    module->struct_count = 3;

    /* Canonical reconstruction retains names, fields and exact bytes. */
    char *text = disasm_module_styled(module, DISASM_STYLE_CANONICAL);
    CHECK(text && strstr(text, ".layouts \""));
    NvmModule *assembled = asm_assemble(text, &result);
    if (!assembled) fprintf(stderr, "%s\n", result.message);
    CHECK(assembled != NULL);
    CHECK(assembled->layout_size == module->layout_size);
    CHECK(memcmp(assembled->layout_data, module->layout_data, module->layout_size) == 0);
    size_t again_size;
    uint8_t *again = serialize(assembled, &again_size);
    CHECK(again_size == size && memcmp(bytes, again, size) == 0);
    free(again); free(text); nvm_module_free(assembled);

    /* Both bridge directions own their copies independently. */
    NvmV2Module decoded;
    CHECK(nvm_v2_module_deserialize(bytes, size, &decoded) == NVM_V2_OK);
    NvmModule *restored = NULL;
    CHECK(nvm_v2_to_nvm_module(&decoded, &restored) == NVM_V2_OK);
    nvm_v2_module_free(&decoded);
    free(bytes); nvm_module_free(module);
    CHECK(nvm_retained_layouts_valid(restored));
    NvmV2Module roundtrip;
    CHECK(nvm_v2_from_nvm_module(restored, &roundtrip) == NVM_V2_OK);
    nvm_module_free(restored);
    CHECK(roundtrip.layouts.count == 3);
    CHECK(roundtrip.layouts.items[0].name_idx == handle);
    CHECK(roundtrip.layouts.items[1].name_idx == other);
    CHECK(roundtrip.layouts.items[2].fields[0].nested_idx == 0);
    CHECK(roundtrip.layouts.items[2].fields[1].nested_idx == 1);
    CHECK(roundtrip.layouts.items[2].fields[1].name_idx == right);
    uint16_t path[] = {1};
    NvmReferencePlace place = {1, 0, 2, path, 1, 1, NVM_REFERENCE_SHARED};
    CHECK(nvm_reference_place_valid(&roundtrip.layouts, 2, &place));
    place.referent_layout = 0;
    CHECK(!nvm_reference_place_valid(&roundtrip.layouts, 2, &place));
    /* Other roundtrip members alias restored's data: only retained layout
     * storage is inspected after its source is freed. */
    nvm_v2_module_free(&roundtrip);

    /* Count-only legacy modules still serialize without typed layout facts. */
    module = nvm_module_new(); CHECK(module != NULL);
    module->struct_count = 3;
    CHECK(nvm_v2_from_nvm_module(module, &roundtrip) == NVM_V2_OK);
    CHECK(!(roundtrip.extra_features & NVM_V2_FEATURE_RETAINED_LAYOUTS));
    CHECK(!nvm_layouts_have_facts(&roundtrip.layouts));
    CHECK(nvm_v2_to_nvm_module(&roundtrip, &restored) == NVM_V2_OK);
    CHECK(restored->layout_data == NULL && restored->layout_size == 0);
    uint8_t *legacy = nvm_serialize(restored, &legacy_size);
    CHECK(legacy != NULL && legacy_size > 0);
    free(legacy); nvm_module_free(restored);
    nvm_v2_module_free(&roundtrip); nvm_module_free(module);
    printf("I passed %u retained-layout checks.\n", checks);
    return 0;
}
