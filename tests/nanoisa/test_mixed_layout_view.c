/* I describe pending modules; I never execute them. */
#include "mixed_layout_view.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "isa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
static long budget = -1;
#define CHECK(x) do { checks++; assert(x); } while (0)
void *mixed_test_malloc(size_t n) {
    if (budget == 0) return NULL;
    if (budget > 0) budget--;
    return malloc(n);
}
void *mixed_test_calloc(size_t n, size_t s) {
    if (budget == 0) return NULL;
    if (budget > 0) budget--;
    return calloc(n, s);
}
static void word(uint8_t *p, uint32_t x) {
    for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(x >> (8 * i));
}
static void retain(NvmModule *m, NvmV2Layout *rows, uint32_t n) {
    NvmV2Layouts l = {rows, n};
    CHECK(nvm_retain_layouts(m, &l) == NVM_V2_OK);
}
static void ownership(NvmModule *m, const uint8_t *flags, uint32_t n, unsigned version) {
    uint32_t at = (8 + n + 3) & ~3u;
    free(m->ownership_data);
    m->ownership_size = at + 4 + (m->function_count ? 20 : 0) + (version == 2 ? 12 : 0);
    m->ownership_data = calloc(m->ownership_size, 1); CHECK(m->ownership_data);
    uint8_t *p = m->ownership_data;
    word(p, version); word(p + 4, n); memcpy(p + 8, flags, n);
    word(p + at, m->function_count); at += 4;
    if (m->function_count) {
        p[at] = 1; p[at + 2] = 1; /* One local, one parameter. */
        p[at + 4] = TAG_VOID; word(p + at + 8, NVM_V2_NO_INDEX);
        p[at + 12] = TAG_STRUCT; word(p + at + 16, 0); at += 20;
    }
    if (version == 2) {
        word(p + at, 1); p[at + 4] = 1; /* One path, one numeric field. */
        p[at + 8] = 7;
    }
}
static NvmMixedLayoutView *describe(NvmModule *m, NvmRecordPlanStatus status) {
    NvmMixedLayoutView sentinel = {0}, *v = &sentinel;
    NvmRecordPlanResult r = nvm_describe_mixed_layouts(m, &v);
    CHECK(r.status == status);
    if (status == NVM_RECORD_DESCRIBED) { CHECK(v != &sentinel); return v; }
    CHECK(v == &sentinel); return NULL;
}
static void expect_class(NvmModule *m, uint32_t row, NvmMixedLayoutClass kind) {
    NvmMixedLayoutView *v = describe(m, NVM_RECORD_DESCRIBED);
    CHECK(v->classes[row] == kind);
    if (kind != NVM_MIXED_ORDINARY_STRUCTURAL) CHECK(v->global_to_managed[row] == NVM_V2_NO_INDEX);
    nvm_mixed_layout_view_free(v);
}
static void clean(NvmModule *m) { free(m->layout_data); free(m->ownership_data); }
static void maps_and_failures(void) {
    NvmModule m = {0}; m.struct_count = 4; m.enum_count = 1;
    NvmV2LayoutField integer = {TAG_INT, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX};
    NvmV2LayoutField array = {TAG_ARRAY, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX};
    NvmV2LayoutField child = {TAG_STRUCT, 3, NVM_V2_NO_INDEX};
    NvmV2Layout rows[] = {
        {NVM_V2_LAYOUT_ENUM, 0, NVM_V2_NO_INDEX, NULL},
        {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &integer},
        {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &integer},
        {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &array},
        {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &child}};
    uint8_t flags[] = {0, 1, 3, 1, 1};
    retain(&m, rows, 5); ownership(&m, flags, 5, 2);
    bool needs = false;
    NvmV2Result old = nvm_ownership_contracts_validate(&m, &needs);
    CHECK(old != NVM_V2_OK); /* The old shared authority still rejects ARRAY. */
    NvmRecordPlan *ordinary = NULL;
    CHECK(nvm_describe_managed_records(&m, &ordinary).status == NVM_RECORD_UNRESOLVED);
    uint8_t *before = malloc(m.layout_size), *owned = malloc(m.ownership_size);
    CHECK(before && owned); memcpy(before, m.layout_data, m.layout_size); memcpy(owned, m.ownership_data, m.ownership_size);
    NvmMixedLayoutView *v = describe(&m, NVM_RECORD_DESCRIBED);
    CHECK(v->record_count == 4 && v->managed_count == 1);
    CHECK(v->classes[0] == NVM_MIXED_UNKNOWN && v->classes[1] == NVM_MIXED_ORDINARY_STRUCTURAL);
    CHECK(v->classes[2] == NVM_MIXED_RESOURCE && v->classes[3] == NVM_MIXED_PENDING_ARRAY_PROOF);
    CHECK(v->classes[4] == NVM_MIXED_PENDING_ARRAY_PROOF);
    CHECK(v->global_to_source[0] == NVM_V2_NO_INDEX);
    for (uint32_t i = 0; i < 4; i++) CHECK(v->source_to_global[i] == i + 1 && v->global_to_source[i + 1] == i);
    CHECK(v->managed_to_global[0] == 1 && v->global_to_managed[1] == 0);
    for (uint32_t i = 2; i < 5; i++) CHECK(v->global_to_managed[i] == NVM_V2_NO_INDEX);
    CHECK(v->ownership_size == m.ownership_size && !memcmp(v->ownership, owned, m.ownership_size));
    nvm_mixed_layout_view_free(v);
    unsigned failures = 0, success = 0;
    for (long n = 0; n < 16; n++) {
        NvmMixedLayoutView sentinel = {0}; v = &sentinel; budget = n;
        NvmRecordPlanResult r = nvm_describe_mixed_layouts(&m, &v); budget = -1;
        if (r.status == NVM_RECORD_MEMORY) { CHECK(v == &sentinel); failures++; }
        else { CHECK(r.status == NVM_RECORD_DESCRIBED && v != &sentinel); nvm_mixed_layout_view_free(v); success++; }
        CHECK(!memcmp(before, m.layout_data, m.layout_size) && !memcmp(owned, m.ownership_data, m.ownership_size));
    }
    CHECK(failures >= 7 && success);
    CHECK(nvm_ownership_contracts_validate(&m, &needs) == old);
    /* Unused last-row malformed bytes cannot be hidden by a positive first row. */
    m.layout_data[m.layout_size - 11] = 1; describe(&m, NVM_RECORD_INVALID);
    memcpy(m.layout_data, before, m.layout_size);
    m.ownership_data[m.ownership_size - 1] = 1; describe(&m, NVM_RECORD_INVALID);
    memcpy(m.ownership_data, owned, m.ownership_size);
    m.ownership_data[8] = 4; describe(&m, NVM_RECORD_INVALID);
    memcpy(m.ownership_data, owned, m.ownership_size);
    word(m.ownership_data, 99); describe(&m, NVM_RECORD_INVALID);
    memcpy(m.ownership_data, owned, m.ownership_size);
    uint32_t size = m.ownership_size; m.ownership_size = NVM_MIXED_MAX_OWNERSHIP_BYTES + 1;
    describe(&m, NVM_RECORD_LIMIT); m.ownership_size = size;
    v = describe(&m, NVM_RECORD_DESCRIBED); clean(&m);
    CHECK(v->layouts.items[4].fields[0].nested_idx == 3 && !memcmp(v->ownership, owned, size));
    nvm_mixed_layout_view_free(v); free(before); free(owned);
}
static void borrowed_leaf(void) {
    NvmModule m = {0}; m.struct_count = 2; m.function_count = 1;
    NvmFunctionEntry fn = {0}; fn.local_count = fn.arity = 1; m.functions = &fn;
    uint8_t tag = TAG_STRUCT, *params[] = {&tag}; m.function_param_types = params;
    NvmV2LayoutField leaf = {TAG_INT, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX};
    NvmV2LayoutField nested = {TAG_STRUCT, 0, NVM_V2_NO_INDEX};
    NvmV2Layout rows[] = {{NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &leaf},
                         {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &nested}};
    uint8_t flags[] = {3, 3}; retain(&m, rows, 2); ownership(&m, flags, 2, 1);
    uint32_t at = 16;
    for (unsigned mode = 0; mode <= 2; mode++) {
        m.ownership_data[at + 13] = (uint8_t)mode;
        word(m.ownership_data + at + 16, 0); expect_class(&m, 1, NVM_MIXED_RESOURCE);
        bool needs = false; CHECK(nvm_ownership_contracts_validate(&m, &needs) == NVM_V2_OK && needs);
        word(m.ownership_data + at + 16, 1);
        if (!mode) expect_class(&m, 1, NVM_MIXED_RESOURCE);
        else { describe(&m, NVM_RECORD_INVALID); CHECK(nvm_ownership_contracts_validate(&m, &needs) != NVM_V2_OK); }
    }
    clean(&m);
}
static void incomplete_and_boundaries(void) {
    NvmModule m = {0}; m.struct_count = 2;
    NvmV2LayoutField missing = {TAG_STRUCT, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX};
    NvmV2LayoutField child = {TAG_STRUCT, 0, NVM_V2_NO_INDEX};
    NvmV2Layout rows[] = {{NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &missing},
                         {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, &child}};
    uint8_t flags[] = {0, 1}; retain(&m, rows, 2); ownership(&m, flags, 2, 1);
    expect_class(&m, 0, NVM_MIXED_UNKNOWN); expect_class(&m, 1, NVM_MIXED_UNKNOWN);
    m.ownership_data[8] = 1; describe(&m, NVM_RECORD_INVALID);
    m.ownership_data[8] = 0; m.ownership_data[9] = 3; describe(&m, NVM_RECORD_INVALID);
    free(m.ownership_data); m.ownership_data = NULL; m.ownership_size = 0;
    expect_class(&m, 1, NVM_MIXED_UNKNOWN);
    missing.type_tag = TAG_HASHMAP; retain(&m, rows, 2); ownership(&m, flags, 2, 1);
    expect_class(&m, 1, NVM_MIXED_UNKNOWN);
    m.ownership_data[8] = 1; expect_class(&m, 0, NVM_MIXED_UNKNOWN);
    missing.type_tag = TAG_INT; retain(&m, rows, 2);
    expect_class(&m, 1, NVM_MIXED_ORDINARY_STRUCTURAL);
    bool needs = true; CHECK(nvm_ownership_contracts_validate(&m, &needs) == NVM_V2_OK && !needs);
    NvmRecordPlan *p = NULL; CHECK(nvm_describe_managed_records(&m, &p).status == NVM_RECORD_DESCRIBED);
    nvm_record_plan_free(p);
    m.struct_count++; describe(&m, NVM_RECORD_INVALID); m.struct_count--;
    m.layout_size--; describe(&m, NVM_RECORD_INVALID); m.layout_size++;
    clean(&m);
    describe(NULL, NVM_RECORD_INVALID); CHECK(nvm_describe_mixed_layouts(NULL, NULL).status == NVM_RECORD_INVALID);
    memset(&m, 0, sizeof m); describe(&m, NVM_RECORD_UNRESOLVED);
    nvm_mixed_layout_view_free(NULL);
}
int main(void) {
    maps_and_failures(); borrowed_leaf(); incomplete_and_boundaries();
    printf("%u mixed descriptor checks passed; no module execution\n", checks); return 0;
}
