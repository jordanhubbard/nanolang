/* I check literal and table allocation failures without replacing pool ownership. */
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../src/nanoisa/nvm_format.h"

static void *owned[16];
static int outstanding, reject_allocation, reject_growth, reject_pool, reject_function;
static void remember(void *pointer) {
    if (!pointer) return;
    assert(outstanding < 16);
    owned[outstanding++] = pointer;
}
static void *literal_malloc(size_t size) {
    if (reject_allocation) return NULL;
    void *pointer = malloc(size);
    remember(pointer);
    return pointer;
}
static void *table_realloc(void *pointer, size_t size) {
    if (reject_growth) return NULL;
    int index = 0;
    if (pointer) {
        while (index < outstanding && owned[index] != pointer) index++;
        assert(index < outstanding);
    }
    int had_pointer = pointer != NULL;
    void *grown = realloc(pointer, size);
    if (grown) {
        if (had_pointer) owned[index] = grown;
        else remember(grown);
    }
    return grown;
}
static void literal_free(void *pointer) {
    if (!pointer) return;
    int index = 0;
    while (index < outstanding && owned[index] != pointer) index++;
    assert(index < outstanding);
    owned[index] = owned[--outstanding];
    free(pointer);
}
static uint32_t literal_add_string(NvmModule *module, const char *value, uint32_t length) {
    if (reject_pool) return UINT32_MAX;
    return nvm_add_string(module, value, length);
}
static uint32_t literal_add_function(NvmModule *module, const NvmFunctionEntry *entry) {
    if (reject_function) return UINT32_MAX;
    return nvm_add_function(module, entry);
}
#define nvm_add_function literal_add_function
#define malloc literal_malloc
#define realloc table_realloc
#define free literal_free
#define nvm_add_string literal_add_string
#include "../../src/nanoisa/assembler.c"
#undef nvm_add_function
#undef malloc
#undef realloc
#undef free
#undef nvm_add_string

static void literal_failures(void) {
    AsmState state = {0};
    state.mod = nvm_module_new();
    assert(state.mod);
    AsmResult result = {0};
    reject_allocation = 1;
    assert(!process_line(&state, ".string value \"hello\"", &result));
    assert(result.error == ASM_ERR_MEMORY && !outstanding);
    reject_allocation = 0;
    reject_pool = 1;
    assert(!process_line(&state, ".string value \"hello\"", &result));
    assert(result.error == ASM_ERR_MEMORY && !outstanding);
    reject_pool = 0;
    assert(!process_line(&state, ".string value \"unterminated", &result));
    assert(result.error == ASM_ERR_SYNTAX && !outstanding);
    assert(!process_line(&state, ".string value \"dangling\\", &result));
    assert(result.error == ASM_ERR_SYNTAX && !outstanding);
    reject_growth = 1;
    assert(!process_line(&state, ".string value \"hello\"", &result));
    assert(result.error == ASM_ERR_MEMORY && !outstanding);
    reject_growth = 0;
    assert(process_line(&state, ".string value \"hello\"", &result));
    assert(state.mod->string_count == 1 && state.symbol_count == 1);
    asm_state_cleanup(&state);
    assert(!outstanding);
    nvm_module_free(state.mod);
}
static void table_failures(void) {
    AsmState state = {0};
    AsmResult result = {0};
    char name[40];
    for (uint32_t i = 0; i < 4096; i++) {
        snprintf(name, sizeof(name), "s%u", i);
        assert(add_symbol(&state, SYMBOL_CONSTANT, name, i, &result));
    }
    Symbol *previous = state.symbols;
    reject_growth = 1;
    assert(!add_symbol(&state, SYMBOL_CONSTANT, "more", 4096, &result));
    assert(result.error == ASM_ERR_MEMORY && state.symbols == previous && state.symbol_count == 4096);
    assert(find_symbol(&state, SYMBOL_CONSTANT, "s4095") == 4095);
    assert(!add_symbol(&state, SYMBOL_CONSTANT, "s0", 0, &result));
    assert(result.error == ASM_ERR_DUPLICATE_SYMBOL);
    assert(!add_label(&state, "label", 0, &result));
    assert(result.error == ASM_ERR_MEMORY && state.label_count == 0);
    assert(!add_patch(&state, "label", 0, 0, &result));
    assert(result.error == ASM_ERR_MEMORY && state.patch_count == 0);
    reject_growth = 0;
    assert(add_symbol(&state, SYMBOL_CONSTANT, "more", 4096, &result));
    for (uint32_t i = 0; i < 3000; i++) {
        snprintf(name, sizeof(name), "l%u", i);
        assert(add_label(&state, name, i, &result));
        assert(add_patch(&state, name, i, i, &result));
    }
    assert(!add_label(&state, "l0", 0, &result));
    assert(result.error == ASM_ERR_DUPLICATE_LABEL);
    uint32_t capacity = INT_MAX;
    assert(!reserve_table(NULL, &capacity, INT_MAX, sizeof(Symbol), &result));
    assert(result.error == ASM_ERR_BAD_OPERAND);
    asm_state_cleanup(&state);
    assert(!outstanding);
}
static void passive_allocation_failures(void) {
    AsmState state = {0};
    state.mod = nvm_module_new();
    assert(state.mod);
    AsmResult result = {0};
    reject_growth = 1;
    assert(!passive_word(&state, 2, &result));
    assert(result.error == ASM_ERR_MEMORY && !state.mod->passive_data);
    assert(!state.mod->passive_size && !outstanding);
    reject_growth = 0;
    for (uint32_t i = 0; i < 32; ++i) assert(passive_word(&state, i, &result));
    uint8_t *previous = state.mod->passive_data;
    reject_growth = 1;
    assert(!passive_word(&state, 32, &result));
    assert(result.error == ASM_ERR_MEMORY && state.mod->passive_data == previous);
    assert(state.mod->passive_size == 128 && state.mod->passive_data[124] == 31);
    reject_growth = 0;
    assert(passive_word(&state, 32, &result));
    assert(state.mod->passive_size == 132);
    literal_free(state.mod->passive_data);
    state.mod->passive_data = NULL;
    state.mod->passive_size = 0;
    nvm_module_free(state.mod);
    assert(!outstanding);
}
static void function_failures(void) {
    AsmState state = {0};
    state.mod = nvm_module_new();
    assert(state.mod);
    AsmResult result = {0};
    assert(add_symbol(&state, SYMBOL_FUNCTION, "main", 0, &result));
    reject_pool = 1;
    assert(!process_line(&state, ".function main 0 0 0 int 1", &result));
    assert(result.error == ASM_ERR_MEMORY && result.message[0]);
    assert(!state.in_function && state.mod->function_count == 0);
    reject_pool = 0;
    reject_function = 1;
    result = (AsmResult){0};
    assert(!process_line(&state, ".function main 0 0 0 int 1", &result));
    assert(result.error == ASM_ERR_MEMORY && result.message[0]);
    assert(!state.in_function && state.mod->function_count == 0);
    reject_function = 0;
    state.symbols[0].value = 1;
    result = (AsmResult){0};
    assert(!process_line(&state, ".function main 0 0 0 int 1", &result));
    assert(result.error == ASM_ERR_DUPLICATE_SYMBOL && result.message[0]);
    assert(!state.in_function);
    asm_state_cleanup(&state);
    nvm_module_free(state.mod);
    assert(!outstanding);
}
int main(void) {
    function_failures();
    passive_allocation_failures();
    literal_failures();
    table_failures();
    puts("I passed literal failures, table growth, duplicate distinctions and allocation recovery.");
    return 0;
}
