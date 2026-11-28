/* I append this probe to the actual generated runtime, with test-only
 * lookup/allocation counters and an explicit abort observation boundary. */
#include <assert.h>
static unsigned char root_test_objects[8192];
static nroot_list root_test_fault_list;

static int root_test_ignored(unsigned kind) {
    return kind == 0 || kind == 2 || kind == 9 || kind == 11;
}

static void root_test_all_identities(void) {
    nroot_list list = {0};
    for (unsigned kind = 0; kind < 256; ++kind) nroot_add(&list, (uint8_t)kind, NULL);
    assert(list.count == 0);
    for (unsigned kind = 0; kind < 256; ++kind)
        for (size_t i = 0; i < 8; ++i) {
            nroot_add(&list, (uint8_t)kind, &root_test_objects[i]);
            nroot_add(&list, (uint8_t)kind, &root_test_objects[i]);
        }
    assert(list.count == 252 * 8);
    size_t position = 0;
    for (unsigned kind = 0; kind < 256; ++kind)
        for (size_t i = 0; i < 8; ++i) {
            assert(nroot_contains(&list, (uint8_t)kind, &root_test_objects[i]) == !root_test_ignored(kind));
            if (!root_test_ignored(kind)) {
                assert(list.items[position].kind == kind && list.items[position].ptr == &root_test_objects[i]);
                ++position;
            }
        }
    assert(position == list.count);
    nroot_reset(&list);
    assert(!list.count && !nroot_contains(&list, 1, &root_test_objects[0]));
    nroot_destroy(&list);
    assert(!list.items && !list.slots && !list.count && !list.capacity && !list.slot_count);
}

static void root_test_one_allocation_abort(size_t n, size_t fault) {
    for (size_t i = 0; i < n; ++i) nroot_add(&root_test_fault_list, 1, &root_test_objects[i]);
    volatile size_t before = root_test_allocations;
    if (n) {
        root_test_fail_at = before + 1;
        nroot_add(&root_test_fault_list, 1, &root_test_objects[0]);
        assert(root_test_allocations == before && root_test_fault_list.count == n);
    }
    root_test_fail_at = before + fault;
    root_test_abort_armed = 1;
    if (!setjmp(root_test_jump)) {
        nroot_add(&root_test_fault_list, 1, &root_test_objects[n]);
        abort();
    }
    root_test_abort_armed = 0;
    root_test_fail_at = 0;
    assert(root_test_allocations == before + fault && root_test_fault_list.count == n);
    for (size_t i = 0; i < n; ++i) {
        assert(root_test_fault_list.items[i].kind == 1 && root_test_fault_list.items[i].ptr == &root_test_objects[i]);
        assert(nroot_contains(&root_test_fault_list, 1, &root_test_objects[i]));
    }
    assert(!nroot_contains(&root_test_fault_list, 1, &root_test_objects[n]));
    /* My real runtime aborts. Only this fixture resumes after its hook. */
    nroot_add(&root_test_fault_list, 1, &root_test_objects[n]);
    assert(root_test_fault_list.count == n + 1 && nroot_contains(&root_test_fault_list, 1, &root_test_objects[n]));
    nroot_destroy(&root_test_fault_list);
}

static void root_test_allocation_aborts(void) {
    const size_t sizes[] = {0, 16, 32, 64};
    for (size_t trial = 0; trial < sizeof sizes / sizeof *sizes; ++trial)
        for (size_t fault = 1; fault <= 2; ++fault)
            root_test_one_allocation_abort(sizes[trial], fault);
}

static int root_test_work(size_t n, int strict) {
    nroot_list list = {0};
    root_test_lookup_calls = 0;
    root_test_allocations = 0;
    for (size_t i = 0; i < n; ++i) nroot_add(&list, 1, &root_test_objects[i]);
    size_t fresh = root_test_lookup_calls, allocations = root_test_allocations;
    size_t expected_allocations = 2;
    for (size_t capacity = 16; capacity < n; capacity *= 2) expected_allocations += 2;
    assert(allocations == expected_allocations);
    for (size_t i = 0; i < allocations; ++i) assert(root_test_allocation_kinds[i] == (i % 2 ? 'A' : 'I'));
    assert(list.count == n && list.capacity == n && list.slot_count == n * 2);
    for (size_t i = 0; i < n; ++i) {
        assert(list.items[i].kind == 1 && list.items[i].ptr == &root_test_objects[i]);
        assert(nroot_contains(&list, 1, &root_test_objects[i]));
    }
    root_test_lookup_calls = 0;
    for (size_t i = 0; i < n; ++i) nroot_add(&list, 1, &root_test_objects[i]);
    size_t duplicates = root_test_lookup_calls;
    assert(duplicates == n && root_test_allocations == allocations && list.count == n);
    nroot_reset(&list);
    assert(!list.count && !nroot_contains(&list, 1, &root_test_objects[0]));
    root_test_lookup_calls = 0;
    root_test_allocations = 0;
    for (size_t i = 0; i < n; ++i) nroot_add(&list, 1, &root_test_objects[i]);
    size_t reused = root_test_lookup_calls;
    assert(!root_test_allocations && list.count == n);
    for (size_t i = 0; i < n; ++i) assert(nroot_contains(&list, 1, &root_test_objects[i]));
    printf("{\"roots\":%zu,\"fresh_lookups\":%zu,\"reused_lookups\":%zu,\"duplicate_lookups\":%zu,\"allocations\":%zu}\n", n, fresh, reused, duplicates, allocations);
    nroot_destroy(&list);
    return strict && (fresh > 2 * n || reused != n);
}

int main(int argc, char **argv) {
    int strict = 1;
    if (argc == 2 && !strcmp(argv[1], "--measure-work")) strict = 0;
    else if (argc != 1) return 2;
    int excessive = 0;
    const size_t sizes[] = {16, 32, 1024, 4096};
    for (size_t i = 0; i < sizeof sizes / sizeof *sizes; ++i) excessive |= root_test_work(sizes[i], strict);
    root_test_all_identities();
    root_test_allocation_aborts();
    puts("I preserved all kinds, order, membership, duplicates, reset, destroy and eight allocation-abort boundaries.");
    return excessive ? 7 : 0;
}
