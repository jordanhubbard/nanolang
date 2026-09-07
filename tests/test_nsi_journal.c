#include "nsi_journal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_sha256_empty(void) {
    const char *test_name = "journal: SHA-256 empty string";
    char hex[NL_JOURNAL_HASH_HEX];
    nl_sha256_hex((const unsigned char *)"", 0, hex);
    if (strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") != 0) {
        FAIL(test_name, hex);
        return;
    }
    PASS(test_name);
}

static int rec(NlJournal *j, NlJournalKind kind, const char *trap, const char *cap,
               const char *method, const char *payload, const char *result) {
    return nl_journal_record(j, kind, trap, cap, method, payload, result,
                             "string", 10, 1, "0");
}

static void test_record_replay_kinds(void) {
    const char *test_name = "journal: record kinds and replay without the service";
    NlJournal *j = nl_journal_create();
    char out[128];
    if (!j) { FAIL(test_name, "create"); return; }
    rec(j, NL_JKIND_TIME, "TRAP_EXTERN_CALL", "cap:time", "nsi:nanolang/time#now", "clock", "123");
    rec(j, NL_JKIND_ENTROPY, "TRAP_EXTERN_CALL", "cap:rand", "nsi:nanolang/rand#bytes", "8", "aabb");
    rec(j, NL_JKIND_FILE, "TRAP_EXTERN_CALL", "cap:nanolang/filesystem.open",
        "nsi:nanolang/filesystem#open", "/tmp/x", "fd:3");
    rec(j, NL_JKIND_NETWORK, "TRAP_EXTERN_CALL", "cap:net", "nsi:nanolang/net#connect", "host", "ok");
    rec(j, NL_JKIND_USER_INPUT, "TRAP_EXTERN_CALL", "cap:input", "nsi:nanolang/input#read", "key", "a");
    rec(j, NL_JKIND_PROCESS, "TRAP_EXTERN_CALL", "cap:proc", "nsi:nanolang/process#spawn", "bin", "pid:1");
    rec(j, NL_JKIND_GPU, "TRAP_EXTERN_CALL", "cap:gpu", "nsi:nanolang/gpu#submit", "q", "ok");
    rec(j, NL_JKIND_AUDIO, "TRAP_EXTERN_CALL", "cap:audio", "nsi:nanolang/audio#write", "pcm", "ok");
    rec(j, NL_JKIND_SERVICE, "TRAP_PRINT", "cap:nanolang/log.write",
        "nsi:nanolang/log#write", "hello-secret", "logged");
    if (nl_journal_count(j) != 9) { FAIL(test_name, "count"); nl_journal_destroy(j); return; }
    if (nl_journal_validate(j) != NL_JOURNAL_OK) { FAIL(test_name, "validate"); nl_journal_destroy(j); return; }
    if (nl_journal_replay(j, 8, out, sizeof(out)) != NL_JOURNAL_OK || strcmp(out, "logged") != 0) {
        FAIL(test_name, "replay");
        nl_journal_destroy(j);
        return;
    }
    PASS(test_name);
    nl_journal_destroy(j);
}

static void test_validate_and_fault(void) {
    const char *test_name = "journal: order, arg hash, schema, fault injection";
    NlJournal *j = nl_journal_create();
    char out[32];
    rec(j, NL_JKIND_SERVICE, "TRAP_PRINT", "cap:nanolang/log.write",
        "nsi:nanolang/log#write", "hi", "ok");
    if (nl_journal_inject_fault(j, 0) != NL_JOURNAL_OK) {
        FAIL(test_name, "inject");
        nl_journal_destroy(j);
        return;
    }
    if (nl_journal_replay(j, 0, out, sizeof(out)) != NL_JOURNAL_ERR_FAULT) {
        FAIL(test_name, "fault-replay");
        nl_journal_destroy(j);
        return;
    }
    PASS(test_name);
    nl_journal_destroy(j);
}

static void test_mock_checkpoint(void) {
    const char *test_name = "journal: mock, checkpoint, seek";
    NlJournal *j = nl_journal_create();
    uint32_t cp = 99;
    char out[32];
    nl_journal_mock(j, "nsi:nanolang/log#write", "mocked");
    rec(j, NL_JKIND_SERVICE, "TRAP_PRINT", "cap:nanolang/log.write",
        "nsi:nanolang/log#write", "x", "live");
    rec(j, NL_JKIND_TIME, "TRAP_EXTERN_CALL", "cap:time", "nsi:nanolang/time#now",
        "clock", "999");
    if (nl_journal_replay(j, 0, out, sizeof(out)) != NL_JOURNAL_OK || strcmp(out, "mocked") != 0) {
        FAIL(test_name, "mock");
        nl_journal_destroy(j);
        return;
    }
    nl_journal_checkpoint(j, &cp);
    if (cp != 0) { FAIL(test_name, "checkpoint"); nl_journal_destroy(j); return; }
    nl_journal_seek(j, 1);
    if (nl_journal_replay(j, 1, out, sizeof(out)) != NL_JOURNAL_OK || strcmp(out, "999") != 0) {
        FAIL(test_name, "seek");
        nl_journal_destroy(j);
        return;
    }
    PASS(test_name);
    nl_journal_destroy(j);
}

static void test_sign_redact_seal(void) {
    const char *test_name = "journal: hash, HMAC, redact export, xor seal";
    NlJournal *j = nl_journal_create();
    char hex[NL_JOURNAL_HASH_HEX];
    char *exp;
    rec(j, NL_JKIND_SERVICE, "TRAP_PRINT", "cap:nanolang/log.write",
        "nsi:nanolang/log#write", "secret-payload", "logged");
    if (nl_journal_hash(j, hex) != NL_JOURNAL_OK || strlen(hex) != 64) {
        FAIL(test_name, "hash");
        nl_journal_destroy(j);
        return;
    }
    if (nl_journal_sign(j, "deploy-key") != NL_JOURNAL_OK ||
        nl_journal_verify(j, "deploy-key") != NL_JOURNAL_OK ||
        nl_journal_verify(j, "wrong") == NL_JOURNAL_OK) {
        FAIL(test_name, "hmac");
        nl_journal_destroy(j);
        return;
    }
    exp = nl_journal_export_redacted(j);
    if (!exp || strstr(exp, "secret-payload") != NULL || !strstr(exp, "arg_hash")) {
        FAIL(test_name, "export");
        free(exp);
        nl_journal_destroy(j);
        return;
    }
    free(exp);
    nl_journal_redact(j, 0);
    if (nl_journal_event(j, 0)->payload[0] != 0) {
        FAIL(test_name, "redact");
        nl_journal_destroy(j);
        return;
    }
    {
        char replay[32];
        rec(j, NL_JKIND_FILE, "TRAP_EXTERN_CALL", "cap:fs", "open", "path-secret", "fd");
        nl_journal_seal(j, "k");
        if (nl_journal_record(j, NL_JKIND_TIME, "TRAP_NONE", "", "", "", "", "string", 0, 0, "0")
            != NL_JOURNAL_ERR_SEALED) {
            FAIL(test_name, "sealed-record");
            nl_journal_destroy(j);
            return;
        }
        if (nl_journal_replay(j, 1, replay, sizeof(replay)) != NL_JOURNAL_OK ||
            strcmp(replay, "fd") != 0) {
            FAIL(test_name, "replay-sealed");
            nl_journal_destroy(j);
            return;
        }
        nl_journal_open(j, "k");
        if (strcmp(nl_journal_event(j, 1)->payload, "path-secret") != 0) {
            FAIL(test_name, "open");
            nl_journal_destroy(j);
            return;
        }
    }
    PASS(test_name);
    nl_journal_destroy(j);
}

int main(void) {
    printf("NSI trap journal tests\n");
    test_sha256_empty();
    test_record_replay_kinds();
    test_validate_and_fault();
    test_mock_checkpoint();
    test_sign_redact_seal();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
