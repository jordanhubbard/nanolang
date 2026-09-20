#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#include "nsi_file_publish.h"
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* I emit JSON code points0..255 as a byte representation, not UTF-8 decoding. */
static void cli_bytes(FILE *out, const char *bytes) {
    if (!bytes) { fputs("null", out); return; }
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)bytes; *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', out); fputc(*p, out); }
        else if (*p < 32 || *p >= 127) fprintf(out, "\\u%04x", (unsigned)*p);
        else fputc(*p, out);
    }
    fputc('"', out);
}
static int cli_report(const NlFileBindingPublishReport *r, const char *input,
                      const char *directory, int result) {
    fprintf(stderr, "{\"status\":%u,\"failed_stage\":%u,\"first_errno\":%d,"
        "\"cleanup_errno\":%d,\"published\":%s,\"durable\":%s,"
        "\"cleanup_pending\":%s,\"path_encoding\":\"byte-codepoints-0-255\",\"input\":",
        (unsigned)r->status, (unsigned)r->failed_stage, r->first_errno,
        r->cleanup_errno, r->published ? "true" : "false",
        r->durable ? "true" : "false", r->cleanup_pending ? "true" : "false");
    cli_bytes(stderr, input); fputs(",\"directory\":", stderr);
    cli_bytes(stderr, directory); fputs(",\"staging_name\":", stderr);
    cli_bytes(stderr, r->staging_name); fputs("}\n", stderr);
    if (fflush(stderr) != 0 || ferror(stderr)) return 1;
    return result;
}
static void cli_fail(NlFileBindingPublishReport *r, NlFileBindingStatus status,
                     int error) {
    if (r->status == NL_FILE_BINDING_OK) {
        r->status = status; r->failed_stage = NL_FILE_PUBLISH_VALIDATE;
        r->first_errno = error;
    }
}
int main(int argc, char **argv) {
    NlFileBindingPublishReport report;
    memset(&report, 0, sizeof report);
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        int rc = fputs("usage: nsi-file-binding INPUT --file-binding-dir DIRECTORY\n", stderr);
        return rc == EOF || fflush(stderr) != 0 ? 1 : 0;
    }
    if (argc != 4 || strcmp(argv[2], "--file-binding-dir")) {
        cli_fail(&report, NL_FILE_BINDING_INVALID, 0);
        return cli_report(&report, NULL, NULL, 2);
    }
    const char *input = argv[1], *directory = argv[3];
    if (strnlen(input, 4096) == 4096 || strnlen(directory, 4096) == 4096) {
        cli_fail(&report, NL_FILE_BINDING_LIMIT, 0);
        return cli_report(&report, NULL, NULL, 1);
    }
    size_t bound = 0;
    if (!nl_file_binding_allocation_bound(&bound) ||
        bound > NL_FILE_BINDING_MAX_ALLOCATION - (NL_FILE_BINDING_MAX_BYTES + 1u)) {
        cli_fail(&report, NL_FILE_BINDING_LIMIT, 0);
        return cli_report(&report, input, directory, 1);
    }
    int fd = open(input, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        cli_fail(&report, NL_FILE_BINDING_IO, errno);
        return cli_report(&report, input, directory, 1);
    }
    unsigned char *bytes = NULL;
    NlFileBindingPlan *plan = NULL;
    struct stat identity;
    size_t used = 0;
    if (fstat(fd, &identity) != 0) cli_fail(&report, NL_FILE_BINDING_IO, errno);
    else if (!S_ISREG(identity.st_mode)) cli_fail(&report, NL_FILE_BINDING_INVALID, 0);
    else if (identity.st_size < 0 || (uintmax_t)identity.st_size > NL_FILE_BINDING_MAX_BYTES)
        cli_fail(&report, NL_FILE_BINDING_LIMIT, 0);
    else {
        bytes = malloc(NL_FILE_BINDING_MAX_BYTES + 1u);
        if (!bytes) cli_fail(&report, NL_FILE_BINDING_MEMORY, 0);
    }
    unsigned interruptions = 0;
    while (report.status == NL_FILE_BINDING_OK) {
        size_t remaining = NL_FILE_BINDING_MAX_BYTES + 1u - used;
        ssize_t count = read(fd, bytes + used, remaining);
        if (count < 0) {
            int error = errno;
            if (error == EINTR && interruptions++ < 64u) continue;
            cli_fail(&report, NL_FILE_BINDING_IO, error);
        } else if (!count) break;
        else if ((size_t)count > remaining) cli_fail(&report, NL_FILE_BINDING_IO, EIO);
        else {
            used += (size_t)count;
            if (used > NL_FILE_BINDING_MAX_BYTES) cli_fail(&report, NL_FILE_BINDING_LIMIT, 0);
        }
    }
    /* I never retry an ambiguous close, even while another error is primary. */
    if (close(fd) != 0) {
        int error = errno;
        if (report.status == NL_FILE_BINDING_OK) cli_fail(&report, NL_FILE_BINDING_IO, error);
        else report.cleanup_errno = error;
        report.cleanup_pending = true;
    }
    if (report.status == NL_FILE_BINDING_OK) {
        NlFileBindingStatus status = nl_file_binding_prepare(bytes, used, &plan);
        if (status != NL_FILE_BINDING_OK) cli_fail(&report, status, 0);
    }
    free(bytes);
    if (report.status == NL_FILE_BINDING_OK)
        nl_file_binding_publish(plan, directory, &report);
    nl_file_binding_free(plan);
    return cli_report(&report, input, directory, report.status == NL_FILE_BINDING_OK ? 0 : 1);
}
