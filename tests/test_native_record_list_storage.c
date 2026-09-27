/* I instrument only the actual shared native list headers in this translation unit. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <limits.h>

static size_t checks, attempts, fail_at = SIZE_MAX, failures, live;
static void *allocated[256];
static size_t allocation_bytes[256];
static bool transient, catch_exit;
static int exit_status;
static jmp_buf terminal;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "I failed %s at %d\n", #x, __LINE__); exit(90); } } while (0)
static bool denied(void) {
    size_t at = attempts++;
    bool fail = transient ? at == fail_at : at >= fail_at;
    if (fail) ++failures;
    return fail;
}
static size_t allocation_slot(void *pointer) {
    for (size_t i = 0; i < 256; ++i) if (allocated[i] == pointer) return i;
    CHECK(false); return 0;
}
static void *observe_malloc(size_t bytes) {
    if (denied()) return NULL;
    void *pointer = malloc(bytes);
    CHECK(pointer != NULL);
    size_t slot = allocation_slot(NULL);
    allocated[slot] = pointer; allocation_bytes[slot] = bytes;
    memset(pointer, 0xa5, bytes); ++live;
    return pointer;
}
static void *observe_realloc(void *pointer, size_t bytes) {
    if (denied()) return NULL;
    size_t slot = allocation_slot(pointer);
    void *result = realloc(pointer, bytes);
    CHECK(result != NULL); allocated[slot] = result;
    if (bytes > allocation_bytes[slot]) memset((unsigned char *)result + allocation_bytes[slot], 0xa5, bytes - allocation_bytes[slot]);
    allocation_bytes[slot] = bytes;
    return result;
}
static void observe_free(void *pointer) {
    if (!pointer) return;
    size_t slot = allocation_slot(pointer);
    allocated[slot] = NULL; allocation_bytes[slot] = 0;
    CHECK(live > 0); --live; free(pointer);
}
static void observe_exit(int status) {
    if (!catch_exit) exit(status);
    exit_status = status;
    longjmp(terminal, 1);
}
#define malloc observe_malloc
#define realloc observe_realloc
#define free observe_free
#define exit observe_exit
#include "../src/runtime/native_record_list.h"
typedef struct { int64_t number; const char *text; } Inner;
typedef struct { Inner inner; int64_t serial; } Item;
typedef struct List_Item List_Item;
NL_DEFINE_RECORD_LIST(Item, Item)
#undef malloc
#undef realloc
#undef free
#undef exit

static Item item(int64_t n) { return (Item){{n, "retained text"}, n + 100}; }
static List_Item *current;
static Item retained;
static Item *saved_data;
static int saved_count, saved_capacity;
static Item saved[128];
static unsigned char saved_bytes[128 * sizeof(Item)];
static bool growing;

static void snapshot(void) {
    saved_data = current->data; saved_count = current->count; saved_capacity = current->capacity;
    CHECK(current->capacity <= 128);
    memcpy(saved_bytes, current->data, (size_t)current->capacity * sizeof(Item));
    for (int i = 0; i < saved_count; ++i) saved[i] = current->data[i];
}
static void unchanged(void) {
    CHECK(current->data == saved_data && current->count == saved_count && current->capacity == saved_capacity);
    CHECK(!memcmp(saved_bytes, current->data, (size_t)saved_capacity * sizeof(Item)));
    for (int i = 0; i < saved_count; ++i) {
        CHECK(current->data[i].inner.number == saved[i].inner.number);
        CHECK(current->data[i].inner.text == saved[i].inner.text);
        CHECK(current->data[i].serial == saved[i].serial);
    }
}
static void reset(void) {
    CHECK(live == 0); attempts = failures = 0; fail_at = SIZE_MAX;
    current = NULL; growing = false; exit_status = 0;
}
static void sequence(void) {
    current = nl_list_Item_new();
    CHECK(nl_list_Item_is_empty(current));
    CHECK(nl_list_Item_capacity(current) == 4);
    for (int i = 0; i < 65; ++i) {
        snapshot(); growing = true;
        if ((i & 1) || i == 4) nl_list_Item_insert(current, 0, item(i));
        else nl_list_Item_push(current, item(i));
        growing = false;
    }
    CHECK(nl_list_Item_length(current) == 65);
    retained = nl_list_Item_remove(current, 0);
    CHECK(retained.inner.number == 63);
    CHECK(current->data[current->count].inner.text == NULL);
    CHECK(current->data[current->count].serial == 0);
    CHECK(nl_list_Item_pop(current).inner.number == 64);
    nl_list_Item_set(current, 0, item(999));
    CHECK(nl_list_Item_get(current, 0).inner.number == 999);
    nl_list_Item_insert(current, current->count, item(777));
    CHECK(nl_list_Item_pop(current).serial == 877);
    CHECK(!strcmp(retained.inner.text, "retained text"));
    int old_count = current->count;
    nl_list_Item_clear(current);
    CHECK(nl_list_Item_is_empty(current));
    for (int i = 0; i < old_count; ++i) CHECK(current->data[i].inner.text == NULL && current->data[i].serial == 0);
    nl_list_Item_push(current, retained);
    CHECK(nl_list_Item_get(current, 0).inner.number == 63);
    nl_list_Item_free(current); current = NULL;
    nl_list_Item_free(NULL);
    CHECK(live == 0);
    CHECK(!strcmp(retained.inner.text, "retained text"));
}
static void allocation_sweep(void) {
    reset(); sequence(); size_t measured = attempts;
    CHECK(measured >= 7);
    for (int mode = 0; mode < 2; ++mode) {
        for (size_t failure = 0; failure < measured; ++failure) {
            reset(); transient = mode != 0; fail_at = failure; catch_exit = true;
            if (!setjmp(terminal)) { sequence(); CHECK(false); }
            catch_exit = false;
            CHECK(exit_status == 1 && failures == 1);
            if (current) {
                CHECK(growing); unchanged();
                fail_at = SIZE_MAX; nl_list_Item_free(current); current = NULL;
            }
            CHECK(live == 0);
            reset(); sequence(); CHECK(attempts == measured);
        }
    }
    printf("I checked %zu allocation positions in prefix and transient modes.\n", measured);
}
static void refuse(const char *mode) {
    if (!strcmp(mode, "null")) { (void)nl_list_Item_get(NULL, 0); return; }
    current = nl_list_Item_new(); nl_list_Item_push(current, item(1)); snapshot();
    if (!strcmp(mode, "negative")) (void)nl_list_Item_get(current, INT64_MIN);
    else if (!strcmp(mode, "huge")) nl_list_Item_set(current, INT64_MAX, item(2));
    else if (!strcmp(mode, "remove-end")) (void)nl_list_Item_remove(current, 1);
    else if (!strcmp(mode, "insert-past")) nl_list_Item_insert(current, 2, item(3));
    else if (!strcmp(mode, "pop-empty")) { nl_list_Item_clear(current); snapshot(); (void)nl_list_Item_pop(current); }
    else if (!strcmp(mode, "narrow-index")) (void)nl_native_list_index(INT64_C(4294967296), 1, false);
    else if (!strcmp(mode, "narrow-capacity")) (void)nl_native_list_capacity(INT64_MAX);
    else if (!strcmp(mode, "negative-capacity")) (void)nl_native_list_capacity(-1);
    else CHECK(false);
}
static void boundary_controls(void) {
    const char *modes[] = {"null", "negative", "huge", "remove-end", "insert-past", "pop-empty",
        "narrow-index", "narrow-capacity", "negative-capacity"};
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        reset(); catch_exit = true;
        if (!setjmp(terminal)) { refuse(modes[i]); CHECK(false); }
        catch_exit = false; CHECK(exit_status == 1);
        if (current) { unchanged(); nl_list_Item_free(current); current = NULL; }
        CHECK(live == 0);
    }
    CHECK(nl_native_list_index(INT_MAX, INT_MAX, true) == INT_MAX);
    CHECK(nl_native_list_capacity(INT_MAX) == INT_MAX);
}
int main(int argc, char **argv) {
    if (argc == 2) {
        /* I use the real exit path after explicit cleanup, so leak detection
         * does not confuse intentional process failure with a leaked fixture. */
        if (!strcmp(argv[1], "real-index")) (void)nl_native_list_index(INT64_MAX, 1, false);
        else if (!strcmp(argv[1], "real-capacity")) (void)nl_native_list_capacity(-1);
        else if (!strcmp(argv[1], "real-null")) (void)nl_list_Item_get(NULL, 0);
        else if (!strcmp(argv[1], "real-oom")) { fail_at = 0; (void)nl_list_Item_new(); }
        else return 91;
        return 92;
    }
    allocation_sweep(); boundary_controls();
    printf("I passed %zu native record-list assertions.\n", checks);
    return 0;
}
