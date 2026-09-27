/* I expose mutable borrowed 0/1/2-string results and independent owned cleanup. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int64_t calls, releases;
static char zero[64], one[32768], two[32768];
int64_t artifact_calls(void) { return calls; }
int64_t artifact_releases(void) { return releases; }
const char *nlc_runtime_root(void) {
    snprintf(zero, sizeof zero, "zero-%lld", (long long)++calls);
    return zero;
}
const char *nlc_module_artifact(const char *value) {
    ++calls;
    if (!strcmp(value, "null")) return NULL;
    snprintf(one, sizeof one, "one-%s", value);
    return one;
}
const char *nl_fs_join_path(const char *left, const char *right) {
    ++calls;
    if (!strcmp(left, "null")) return NULL;
    snprintf(two, sizeof two, "two-%s:%s", left, right);
    return two;
}
const char *nl_nanoisa_last_error(void) {
    ++calls;
    char *text = malloc(6);
    if (text) memcpy(text, "owned", 6);
    return text;
}
void nl_nanoisa_last_error__nano_string_release_v1(const char *text) {
    ++releases;
    free((void *)text);
}
