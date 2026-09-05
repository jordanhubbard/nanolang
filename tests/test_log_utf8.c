#include "../modules/std/log/log.h"
#include "utf8.h"
#include "diag_id.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static int slurp(const char *path, char *buf, size_t n) {
    FILE *fp = fopen(path, "rb");
    size_t got;
    if (!fp) return 0;
    got = fread(buf, 1, n - 1, fp);
    fclose(fp);
    buf[got] = '\0';
    return 1;
}

static void test_invalid_utf8_marker(void) {
    const char *test_name = "log: invalid UTF-8 becomes marker and keeps LOG01";
    const char *path = "/tmp/nl_log_utf8.txt";
    char junk[] = { 'h', 'i', (char)0xFF, 0 };
    char buf[512];

    unlink(path);
    nl_log_set_level(1);
    nl_log_set_output_mode(1);
    nl_log_set_file(path);
    nl_log_write(1, junk);
    nl_log_cleanup();
    if (!slurp(path, buf, sizeof buf))
        { FAIL(test_name, "read"); return; }
    if (!strstr(buf, "LOG01"))
        { FAIL(test_name, buf); return; }
    if (!strstr(buf, "<invalid UTF-8>"))
        { FAIL(test_name, buf); return; }
    unlink(path);
    PASS(test_name);
}

static void test_bidi_and_ansi(void) {
    const char *test_name = "log: strip bidi override and ANSI CSI";
    const char *path = "/tmp/nl_log_bidi.txt";
    /* U+202E RLO + admin + ESC[31m red */
    char payload[64];
    char buf[512];
    snprintf(payload, sizeof payload, "\xE2\x80\xAE""admin\x1B[31mred");

    unlink(path);
    nl_log_set_level(1);
    nl_log_set_output_mode(1);
    nl_log_set_file(path);
    nl_log_write_event(1, "LOG01", payload);
    nl_log_cleanup();
    if (!slurp(path, buf, sizeof buf))
        { FAIL(test_name, "read"); return; }
    if (strstr(buf, "\xE2\x80\xAE"))
        { FAIL(test_name, "rlo survived"); unlink(path); return; }
    if (strstr(buf, "\x1B"))
        { FAIL(test_name, "esc survived"); unlink(path); return; }
    if (!strstr(buf, "admin") || !strstr(buf, "red"))
        { FAIL(test_name, buf); unlink(path); return; }
    unlink(path);
    PASS(test_name);
}

int main(void) {
    printf("Log UTF-8 and event-id tests\n");
    test_invalid_utf8_marker();
    test_bidi_and_ansi();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
