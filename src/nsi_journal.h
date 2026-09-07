#ifndef NL_NSI_JOURNAL_H
#define NL_NSI_JOURNAL_H

#include <stddef.h>
#include <stdint.h>

#define NL_JOURNAL_VERSION 0
#define NL_JOURNAL_OK 0
#define NL_JOURNAL_ERR 1
#define NL_JOURNAL_ERR_ORDER 2
#define NL_JOURNAL_ERR_ARG 3
#define NL_JOURNAL_ERR_CAP 4
#define NL_JOURNAL_ERR_SCHEMA 5
#define NL_JOURNAL_ERR_FAULT 6
#define NL_JOURNAL_ERR_SEALED 7
#define NL_JOURNAL_MAX 64
#define NL_JOURNAL_HASH_HEX 65

typedef enum {
    NL_JKIND_TIME = 0,
    NL_JKIND_ENTROPY,
    NL_JKIND_FILE,
    NL_JKIND_NETWORK,
    NL_JKIND_USER_INPUT,
    NL_JKIND_PROCESS,
    NL_JKIND_GPU,
    NL_JKIND_AUDIO,
    NL_JKIND_SERVICE
} NlJournalKind;

typedef struct {
    uint32_t seq;
    NlJournalKind kind;
    char trap[32];
    char capability[64];
    char method[80];
    char arg_hash[NL_JOURNAL_HASH_HEX];
    char payload[128];
    char result[128];
    char result_schema[32];
    uint64_t timing_ns;
    int generation;
    char impl_version[16];
    int redacted;
    int fault;
} NlJournalEvent;

typedef struct NlJournal NlJournal;

NlJournal *nl_journal_create(void);
void nl_journal_destroy(NlJournal *j);

int nl_journal_record(NlJournal *j, NlJournalKind kind, const char *trap,
                      const char *capability, const char *method,
                      const char *payload, const char *result,
                      const char *result_schema, uint64_t timing_ns,
                      int generation, const char *impl_version);

int nl_journal_count(const NlJournal *j);
const NlJournalEvent *nl_journal_event(const NlJournal *j, int i);

int nl_journal_replay(NlJournal *j, uint32_t seq, char *out, size_t n);
int nl_journal_validate(const NlJournal *j);

int nl_journal_mock(NlJournal *j, const char *method, const char *result);
int nl_journal_inject_fault(NlJournal *j, uint32_t seq);
int nl_journal_checkpoint(const NlJournal *j, uint32_t *seq_out);
int nl_journal_seek(NlJournal *j, uint32_t seq);

int nl_journal_hash(const NlJournal *j, char hex[NL_JOURNAL_HASH_HEX]);
int nl_journal_sign(NlJournal *j, const char *key);
int nl_journal_verify(const NlJournal *j, const char *key);

int nl_journal_redact(NlJournal *j, uint32_t seq);
char *nl_journal_export_redacted(const NlJournal *j);
int nl_journal_seal(NlJournal *j, const char *key);
int nl_journal_open(NlJournal *j, const char *key);

void nl_sha256_hex(const unsigned char *data, size_t n, char hex[NL_JOURNAL_HASH_HEX]);

#endif
