#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ownership_contracts.h"
#include "retained_layouts.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "isa.h"

static unsigned checks;
#define CHECK(c) do { checks++; assert(c); } while (0)
static void word(uint8_t *data, unsigned offset, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) data[offset+i] = (uint8_t)(value >> (8*i));
}
static void slot(uint8_t *data, unsigned offset, uint8_t tag, uint8_t mode, uint32_t layout) {
    data[offset] = tag; data[offset+1] = mode; word(data, offset+4, layout);
}
static uint8_t *wire(const NvmModule *module, size_t *size, const char *path) {
    NvmV2Module out;
    CHECK(nvm_v2_from_nvm_module(module, &out) == NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&out, NULL, 0, size) == NVM_V2_OK);
    uint8_t *bytes = malloc(*size); CHECK(bytes != NULL);
    CHECK(nvm_v2_module_serialize(&out, bytes, *size, NULL) == NVM_V2_OK);
    if (path) {
        FILE *file = fopen(path, "wb"); CHECK(file != NULL);
        CHECK(fwrite(bytes, 1, *size, file) == *size); CHECK(fclose(file) == 0);
    }
    nvm_v2_module_free(&out); return bytes;
}
static void check_status(NvmModule *module, bool valid, bool needs) {
    bool found = false;
    NvmV2Result result = nvm_ownership_contracts_validate(module, &found);
    CHECK((result == NVM_V2_OK) == valid);
    if (valid) CHECK(found == needs);
}

int main(int argc, char **argv) {
    AsmResult result;
    NvmModule *module = asm_assemble(
        ".types 1 0 0\n.entry 1\n.function read 1 1 0 int 1\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n.parameters 0 struct\n"
        ".function main 0 1 0 int 1\nPUSH_I64 42\nAGG_PACK 0 0 0 1\n"
        "STORE_LOCAL 0\nLOAD_LOCAL 0\nCALL 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n", &result);
    if (!module) fprintf(stderr, "%s\n", result.message);
    CHECK(module != NULL);
    NvmV2LayoutField field = {TAG_INT, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX};
    NvmV2Layout item = {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &field};
    NvmV2Layouts layouts = {&item, 1};
    CHECK(nvm_retain_layouts(module, &layouts) == NVM_V2_OK);
    module->ownership_size = 56;
    module->ownership_data = calloc(module->ownership_size, 1);
    CHECK(module->ownership_data != NULL);
    uint8_t *data = module->ownership_data;
    word(data, 0, 1); word(data, 4, 1); data[8] = NVM_LAYOUT_COMPLETE;
    word(data, 12, 2);
    data[16] = 1; data[18] = 1; /* read: one local, one parameter */
    slot(data, 20, TAG_INT, 0, NVM_V2_NO_INDEX);
    slot(data, 28, TAG_STRUCT, 0, 0);
    data[36] = 1; /* main: one local, no parameters */
    slot(data, 40, TAG_INT, 0, NVM_V2_NO_INDEX);
    slot(data, 48, TAG_STRUCT, 0, 0);
    check_status(module, true, false);
    CHECK(nvm_verify(module).ok);
    size_t size;
    uint8_t *bytes = wire(module, &size, argc > 1 ? argv[1] : NULL);
    NvmV2Header header;
    CHECK(nvm_v2_read_header(bytes, size, &header) == NVM_V2_OK);
    CHECK(header.feature_bits & NVM_V2_FEATURE_OWNERSHIP);
    free(bytes);
    char *text = disasm_module_styled(module, DISASM_STYLE_CANONICAL);
    CHECK(text && strstr(text, ".ownership \""));
    NvmModule *copy = asm_assemble(text, &result);
    CHECK(copy != NULL);
    CHECK(copy->ownership_size == module->ownership_size);
    CHECK(!memcmp(copy->ownership_data, data, module->ownership_size));
    nvm_module_free(copy); free(text);

    /* Shared/exclusive declarations remain transportable but cannot execute
     * before genuine references and instruction lifetimes are implemented. */
    data[8] |= NVM_LAYOUT_RESOURCE;
    data[29] = 1;
    check_status(module, true, true);
    CHECK(!nvm_verify(module).ok);
    bytes = wire(module, &size, argc > 2 ? argv[2] : NULL);
    NvmV2Module decoded;
    CHECK(nvm_v2_module_deserialize(bytes, size, &decoded) == NVM_V2_OK);
    CHECK(nvm_v2_to_nvm_module(&decoded, &copy) == NVM_V2_OK);
    nvm_v2_module_free(&decoded); free(bytes);
    check_status(copy, true, true);
    CHECK(copy->ownership_data[29] == 1);
    CHECK(copy->ownership_data != module->ownership_data);
    text = disasm_module_styled(copy, DISASM_STYLE_CANONICAL);
    CHECK(text != NULL);
    CHECK(asm_assemble(text, &result) == NULL);
    CHECK(result.error == ASM_ERR_VERIFY);
    NvmModule *retained = asm_assemble_unverified(text, &result);
    CHECK(retained != NULL);
    check_status(retained, true, true);
    CHECK(!memcmp(retained->ownership_data, data, module->ownership_size));
    nvm_module_free(retained); nvm_module_free(copy); free(text);
    data[29] = 2; check_status(module, true, true);

    /* Normal declaration mistakes do not become unknown reference facts. */
    data[29] = 3; check_status(module, false, false); data[29] = 1;
    data[8] = NVM_LAYOUT_COMPLETE; check_status(module, false, false);
    data[8] = NVM_LAYOUT_RESOURCE; check_status(module, false, false);
    data[8] = NVM_LAYOUT_COMPLETE | NVM_LAYOUT_RESOURCE;
    data[49] = 1; check_status(module, false, false); data[49] = 0;
    data[21] = 1; check_status(module, false, false); data[21] = 0;
    data[28] = TAG_INT; check_status(module, false, false); data[28] = TAG_STRUCT;
    word(data, 32, NVM_V2_NO_INDEX); check_status(module, false, false); word(data, 32, 0);
    word(data, 52, 1); check_status(module, false, false); word(data, 52, 0);
    data[18] = 0; check_status(module, false, false); data[18] = 1;
    data[36] = 2; check_status(module, false, false); data[36] = 1;
    data[20] = TAG_BOOL; check_status(module, false, false); data[20] = TAG_INT;
    data[30] = 1; check_status(module, false, false); data[30] = 0;
    data[9] = 1; check_status(module, false, false); data[9] = 0;
    word(data, 0, 2); check_status(module, false, false); word(data, 0, 1);
    word(data, 12, 1); check_status(module, false, false); word(data, 12, 2);
    check_status(module, true, true);
    /* A complete outer record must propagate a nested resource obligation. */
    NvmV2LayoutField nested = {TAG_STRUCT, 0, NVM_V2_NO_INDEX};
    NvmV2Layout pair[] = {item, {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &nested}};
    NvmV2Layouts tree = {pair, 2};
    module->struct_count = 2;
    CHECK(nvm_retain_layouts(module, &tree) == NVM_V2_OK);
    word(data, 4, 2); data[9] = NVM_LAYOUT_COMPLETE;
    check_status(module, false, false);
    data[9] |= NVM_LAYOUT_RESOURCE;
    check_status(module, true, true);
    /* A borrowed leaf cannot silently become an aggregate referent. */
    word(data, 32, 1);
    check_status(module, false, false);
    word(data, 32, 0);
    check_status(module, true, true);
    nvm_module_free(module);
    printf("I passed %u ownership-contract checks.\n", checks);
    return 0;
}
