/* I own semantic keys and short C identifiers for one emission. */
typedef struct { char *key; char *name; } NativeOpaqueName;
typedef struct {
    NativeOpaqueName *rows;
    size_t count, prefix;
} NativeOpaqueNames;
static _Thread_local NativeOpaqueNames *native_opaque_names;
static void native_opaque_name_failure(void) {
    fprintf(stderr, "I cannot retain a complete opaque-bearing native type name\n");
    exit(1);
}
static const char *native_opaque_projection(const char *key) {
    if (!key || strncmp(key, "#otype:", 7)) return key;
    NativeOpaqueNames *table = native_opaque_names;
    if (!table) native_opaque_name_failure();
    for (size_t i = 0; i < table->count; ++i)
        if (!strcmp(table->rows[i].key, key)) return table->rows[i].name;
    if (table->count >= SIZE_MAX / sizeof(NativeOpaqueName)) native_opaque_name_failure();
    char spelling[64 + 6 * sizeof(size_t)];
    int length = snprintf(spelling, sizeof spelling, "__nano_opaque_%zu_%zu", table->prefix, table->count);
    if (length < 0 || (size_t)length >= sizeof spelling) native_opaque_name_failure();
    char *owned_key = strdup(key), *owned_name = strdup(spelling);
    NativeOpaqueName *rows = malloc((table->count + 1) * sizeof *rows);
    if (!owned_key || !owned_name || !rows) {
        free(owned_key); free(owned_name); free(rows); native_opaque_name_failure();
    }
    if (table->count) memcpy(rows, table->rows, table->count * sizeof *rows);
    rows[table->count] = (NativeOpaqueName){owned_key, owned_name};
    free(table->rows); table->rows = rows; ++table->count;
    return owned_name;
}
static void native_opaque_names_free(NativeOpaqueNames *table) {
    for (size_t i = 0; i < table->count; ++i) {
        free(table->rows[i].key); free(table->rows[i].name);
    }
    free(table->rows);
}
