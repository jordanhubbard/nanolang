#include <assert.h>
#include <stdio.h>
#include "reference_places.h"
#include "isa.h"

static unsigned checks;
#define CHECK(condition) do { checks++; assert(condition); } while (0)

int main(void) {
    /* Handle and Other have equal fields but distinct nominal indices. Pair
     * has two independent Handle fields; Wrapper gives a two-level path. */
    NvmV2LayoutField scalars[] = {
        {TAG_INT, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX},
        {TAG_U8, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX},
        {TAG_FLOAT, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX},
        {TAG_BOOL, NVM_V2_NO_INDEX, NVM_V2_NO_INDEX}
    };
    NvmV2LayoutField pair[] = {
        {TAG_STRUCT, 0, NVM_V2_NO_INDEX},
        {TAG_STRUCT, 0, NVM_V2_NO_INDEX}
    };
    NvmV2LayoutField wrapper[] = {{TAG_STRUCT, 2, NVM_V2_NO_INDEX}};
    NvmV2Layout items[] = {
        {NVM_V2_LAYOUT_STRUCT, 4, NVM_V2_NO_INDEX, scalars},
        {NVM_V2_LAYOUT_STRUCT, 4, NVM_V2_NO_INDEX, scalars},
        {NVM_V2_LAYOUT_STRUCT, 2, NVM_V2_NO_INDEX, pair},
        {NVM_V2_LAYOUT_STRUCT, 1, NVM_V2_NO_INDEX, wrapper}
    };
    NvmV2Layouts layouts = {items, 4};
    uint16_t left_path[] = {0, 0}, right_path[] = {0, 1};
    NvmReferencePlace left = {17, 3, 3, left_path, 2, 0, NVM_REFERENCE_SHARED};
    NvmReferencePlace right = {17, 3, 3, right_path, 2, 0, NVM_REFERENCE_EXCLUSIVE};
    NvmReferencePlace root = {17, 3, 0, NULL, 0, 0, NVM_REFERENCE_EXCLUSIVE};
    CHECK(nvm_reference_place_valid(&layouts, 3, &left));
    CHECK(nvm_reference_place_valid(&layouts, 3, &right));
    CHECK(nvm_reference_place_valid(&layouts, 0, &root));
    CHECK(!nvm_reference_place_valid(&layouts, 1, &root));
    root.root_layout = root.referent_layout = 1;
    CHECK(nvm_reference_place_valid(&layouts, 1, &root));
    root.referent_layout = 0;
    CHECK(!nvm_reference_place_valid(&layouts, 1, &root));

    CHECK(!nvm_reference_places_overlap(&left, &right));
    CHECK(!nvm_reference_holds_conflict(&left, &right));
    CHECK(nvm_reference_places_overlap(&left, &left));
    CHECK(!nvm_reference_holds_conflict(&left, &left));
    CHECK(nvm_reference_holds_conflict(&right, &right));
    CHECK(!nvm_reference_owner_access_conflicts(&left, &left, false));
    CHECK(nvm_reference_owner_access_conflicts(&left, &left, true));
    CHECK(nvm_reference_owner_access_conflicts(&right, &right, false));
    CHECK(!nvm_reference_owner_access_conflicts(&right, &left, true));

    /* Prefix access descriptors need not themselves be scalar referents.
     * Relabeling a layout must never make the same owner look disjoint. */
    NvmReferencePlace parent = left;
    parent.field_count = 1;
    parent.referent_layout = 2;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &parent));
    CHECK(nvm_reference_places_overlap(&parent, &right));
    CHECK(nvm_reference_holds_conflict(&parent, &right));
    CHECK(nvm_reference_owner_access_conflicts(&left, &parent, true));
    parent.field_count = 0;
    parent.root_layout = 123;
    CHECK(nvm_reference_places_overlap(&parent, &left));
    CHECK(nvm_reference_owner_access_conflicts(&right, &parent, true));
    parent.invocation++;
    CHECK(!nvm_reference_places_overlap(&parent, &left));
    parent = left;
    parent.local++;
    CHECK(!nvm_reference_places_overlap(&parent, &left));

    /* Descriptor and retained layout rejection are ordinary schema tests;
     * the public API also accepts producer tables before serialization. */
    NvmReferencePlace bad = left;
    bad.referent_layout = 1;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &bad));
    bad = left; bad.root_layout = 4;
    CHECK(!nvm_reference_place_valid(&layouts, 4, &bad));
    bad = left; bad.fields = NULL;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &bad));
    CHECK(nvm_reference_holds_conflict(&left, &bad));
    CHECK(nvm_reference_owner_access_conflicts(&bad, &left, false));
    bad = left; bad.mode = 0;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &bad));
    CHECK(nvm_reference_holds_conflict(&bad, &left));
    bad = left; bad.invocation = 0;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &bad));
    CHECK(nvm_reference_places_overlap(NULL, &left));
    CHECK(!nvm_reference_place_valid(NULL, 3, &left));
    CHECK(!nvm_reference_place_valid(&layouts, 3, NULL));
    left_path[1] = 2;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &left));
    left_path[1] = 0;
    wrapper[0].nested_idx = 3;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &left));
    wrapper[0].nested_idx = NVM_V2_NO_INDEX;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &left));
    wrapper[0].nested_idx = 2;
    wrapper[0].type_tag = TAG_ARRAY;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &left));
    wrapper[0].type_tag = TAG_STRUCT;
    items[2].kind = NVM_V2_LAYOUT_UNION;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &left));
    items[2].kind = NVM_V2_LAYOUT_STRUCT;
    items[2].fields = NULL;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &left));
    items[2].fields = pair;
    scalars[0].type_tag = TAG_STRING;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &left));
    scalars[0].type_tag = TAG_INT;
    scalars[0].nested_idx = 0;
    CHECK(!nvm_reference_place_valid(&layouts, 3, &left));
    scalars[0].nested_idx = NVM_V2_NO_INDEX;
    CHECK(nvm_reference_place_valid(&layouts, 3, &left));

    /* Every pair in this set obeys symmetric overlap and conflict; owner
     * access is deliberately asymmetric with respect to hold mode. */
    NvmReferencePlace places[] = {left, right, parent, root};
    for (unsigned i = 0; i < 4; i++) for (unsigned j = 0; j < 4; j++) {
        CHECK(nvm_reference_places_overlap(&places[i], &places[j]) ==
              nvm_reference_places_overlap(&places[j], &places[i]));
        CHECK(nvm_reference_holds_conflict(&places[i], &places[j]) ==
              nvm_reference_holds_conflict(&places[j], &places[i]));
    }
    printf("I passed %u reference-place checks.\n", checks);
    return 0;
}
