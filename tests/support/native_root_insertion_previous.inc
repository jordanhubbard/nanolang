/* I retain the pre-correction helper as a test-only cost countercase. */
static void nroot_add(nroot_list *list, uint8_t kind, const void *ptr) {
    if (!ptr || kind == 0 || kind == 2 || kind == 9 || kind == 11) return;
    if (nroot_contains(list, kind, ptr)) return;
    if (list->count >= list->slot_count / 2) {
        if (list->slot_count > SIZE_MAX / 2 / sizeof *list->slots) NVM2C_ABORT();
        size_t count = list->slot_count ? list->slot_count * 2 : 32;
        size_t *slots = calloc(count, sizeof *slots);
        if (!slots) NVM2C_ABORT();
        free(list->slots); list->slots = slots; list->slot_count = count;
        for (size_t i = 0; i < list->count; ++i)
            list->slots[nroot_slot(list, list->items[i].kind, list->items[i].ptr)] = i + 1;
    }
    if (list->count == list->capacity) {
        if (list->capacity > SIZE_MAX / 2 / sizeof *list->items) NVM2C_ABORT();
        size_t capacity = list->capacity ? list->capacity * 2 : 16;
        nroot_ref *items = realloc(list->items, capacity * sizeof *items);
        if (!items) NVM2C_ABORT();
        list->items = items; list->capacity = capacity;
    }
    size_t slot = nroot_slot(list, kind, ptr);
    list->items[list->count++] = (nroot_ref){kind, ptr};
    list->slots[slot] = list->count;
}
