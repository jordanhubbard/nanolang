#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "service_shadows.h"
#include "shadow_timeout.h"
#include "../nanoisa/file_indirect_public.h"
#include "../nanoisa/socket_indirect_public.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static bool record(FILE *log, const char *event, size_t index,
                   const NlServiceShadow *shadow) {
    if (fprintf(log, "%s %zu ", event, index) < 0) return false;
    const char *fields[] = {shadow->origin, shadow->name};
    for (unsigned i = 0; i < 2; ++i) {
        for (const unsigned char *p = (const unsigned char *)fields[i]; *p; ++p)
            if (fprintf(log, "%02x", *p) < 0) return false;
        if (fputc(i ? '\n' : ' ', log) == EOF) return false;
    }
    return fflush(log) == 0 && fsync(fileno(log)) == 0;
}

static bool elapsed(struct timespec start, unsigned seconds, bool *failed) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) { *failed = true; return true; }
    return now.tv_sec - start.tv_sec > seconds ||
        (now.tv_sec - start.tv_sec == seconds && now.tv_nsec >= start.tv_nsec);
}

static NlServiceShadowReport execute(const NlServiceShadow *suite, size_t count, unsigned catalog, FILE *log) {
    NlServiceShadowReport report = {NL_SERVICE_SHADOW_FAILED, 0};
    const NvmFileIndirectOptions options = {NVM_FILE_INDIRECT_RUNTIME_REVISION,
                                         NVM_FILE_INDIRECT_FUEL_MAX};
    for (size_t i = 0; i < count; ++i) {
        if (!record(log, "START", i, &suite[i])) return report;
        if(catalog==2) {
        const NvmSocketIndirectOptions tcp_options={1,NVM_SOCKET_INDIRECT_FUEL_MAX};
        NvmSocketHostGrant *grant = NULL;
        if (nvm_socket_host_grant_create_tcp_connections(&grant) != NVM_SOCKET_HOST_OK)
            return report;
        NvmSocketScalar scalar = {0};
        NvmSocketIndirectExecutionReport result = nvm_socket_execute_indirect_bytes(
            grant, suite[i].bytes, suite[i].size, &tcp_options, &scalar);
        NvmSocketHostStatus revoked = nvm_socket_host_grant_revoke(grant);
        NvmSocketHostStatus destroyed = nvm_socket_host_grant_destroy(&grant);
        if (result.runtime.status != NVM_SOCKET_RUNTIME_OK || !result.runtime.acquired ||
            result.runtime.cleanup.cleanup_failures || revoked != NVM_SOCKET_HOST_OK ||
            destroyed != NVM_SOCKET_HOST_OK || grant || scalar.value != 0) return report;
        } else {
        NvmFileHostGrant *grant = NULL;
        if (nvm_file_host_grant_create_temporary_files(&grant) != NVM_FILE_HOST_OK)
            return report;
        NvmFileScalar scalar = {0};
        NvmFileIndirectExecutionReport result = nvm_file_execute_indirect_bytes(
            grant, suite[i].bytes, suite[i].size, &options, &scalar);
        NvmFileHostStatus revoked = nvm_file_host_grant_revoke(grant);
        NvmFileHostStatus destroyed = nvm_file_host_grant_destroy(&grant);
        if (result.runtime.status != NVM_FILE_RUNTIME_OK || !result.runtime.acquired ||
            result.runtime.cleanup.cleanup_failures || revoked != NVM_FILE_HOST_OK ||
            destroyed != NVM_FILE_HOST_OK || grant || scalar.value != 0) return report;
        }
        if (!record(log, "DONE", i, &suite[i])) return report;
        ++report.completed;
    }
    report.status = NL_SERVICE_SHADOW_OK;
    return report;
}

NlServiceShadowReport nl_service_run_catalog_shadows(const NlServiceShadow *suite, size_t count,
                                            unsigned catalog, bool allowed, const char *path) {
    NlServiceShadowReport report = {NL_SERVICE_SHADOW_INVALID, 0};
    if ((catalog!=1 && catalog!=2) || (!suite && count) || count > 4096 || !path || !*path) return report;
    for (size_t i = 0; i < count; ++i)
        if (!suite[i].bytes || !suite[i].size || !suite[i].origin || !suite[i].name ||
            !*suite[i].origin || !*suite[i].name ||
            strnlen(suite[i].origin, 4097) > 4096 || strnlen(suite[i].name, 4097) > 4096)
            return report;
    if (!allowed && count) { report.status = NL_SERVICE_SHADOW_DENIED; return report; }
    int seconds = nl_shadow_timeout_seconds(10);
    if (seconds < 0) return report;
    report.status = NL_SERVICE_SHADOW_SYSTEM;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return report;
    FILE *log = fdopen(fd, "w");
    if (!log) { close(fd); return report; }
    for (size_t i = 0; i < count; ++i)
        if (!record(log, "SELECT", i, &suite[i])) { fclose(log); return report; }
    int channel[2];
    if (pipe(channel)) { fclose(log); return report; }
    if (fcntl(channel[0], F_SETFL, O_NONBLOCK) < 0 ||
        fcntl(channel[0], F_SETFD, FD_CLOEXEC) < 0 ||
        fcntl(channel[1], F_SETFD, FD_CLOEXEC) < 0) {
        close(channel[0]); close(channel[1]); fclose(log); return report;
    }
    struct timespec start;
    if (clock_gettime(CLOCK_MONOTONIC, &start)) {
        close(channel[0]); close(channel[1]); fclose(log); return report;
    }
    fflush(NULL);
    pid_t child = fork();
    int fork_error = errno;
    if (!child) {
        close(channel[0]);
        if (setpgid(0, 0)) {
            dprintf(fd,"I cannot establish my shadow process group: %s.\n",strerror(errno));_exit(1);
        }
        if (dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0) {
            dprintf(fd,"I cannot redirect my shadow log: %s.\n",strerror(errno));_exit(1);
        }
        NlServiceShadowReport result = execute(suite, count, catalog, log);
        if (fclose(log)) result.status = NL_SERVICE_SHADOW_SYSTEM;
        ssize_t sent;
        do { sent = write(channel[1], &result, sizeof result); } while (sent < 0 && errno == EINTR);
        close(channel[1]);
        _exit(sent == sizeof result ? 0 : 1);
    }
    close(channel[1]);
    bool close_failed = fclose(log) != 0;
    if (child < 0) { fprintf(stderr,"I cannot start my shadow child: %s.\n",strerror(fork_error));close(channel[0]);return report; }
    /* Both sides establish the group so termination also reaches descendants. */
    (void)setpgid(child, child);
    int status = 0;
    pid_t waited;
    bool clock_failed = false, timeout = false;
    struct timespec pause = {0, 10000000};
    for (;;) {
        waited = waitpid(child, &status, WNOHANG);
        if (waited == child || (waited < 0 && errno != EINTR)) break;
        if (waited < 0) waited = 0;
        if (elapsed(start, (unsigned)seconds, &clock_failed)) { timeout = true; break; }
        nanosleep(&pause, NULL);
    }
    /* I remove descendants on successful completion as well as failure. */
    bool kill_failed = kill(-child, SIGKILL) != 0 && errno != ESRCH;
    if (waited != child && waited >= 0) {
        if (kill(child, SIGKILL) && errno != ESRCH) kill_failed = true;
        struct timespec cleanup;
        if (clock_gettime(CLOCK_MONOTONIC, &cleanup)) clock_failed = true;
        else for (;;) {
            waited = waitpid(child, &status, WNOHANG);
            if (waited == child || (waited < 0 && errno != EINTR)) break;
            if (waited < 0) waited = 0;
            if (elapsed(cleanup, 1, &clock_failed)) break;
            nanosleep(&pause, NULL);
        }
    }
    NlServiceShadowReport received = {NL_SERVICE_SHADOW_SYSTEM, 0};
    ssize_t got;
    do { got = read(channel[0], &received, sizeof received); } while (got < 0 && errno == EINTR);
    close(channel[0]);
    if (close_failed || kill_failed || clock_failed || waited != child) {
        fprintf(stderr,"I cannot finish shadow supervision (log %u, cleanup %u, clock %u, waited %ld, child %ld).\n",
            (unsigned)close_failed,(unsigned)kill_failed,(unsigned)clock_failed,(long)waited,(long)child);return report;
    }
    if (timeout) { report.status = NL_SERVICE_SHADOW_TIMEOUT; return report; }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || got != sizeof received) {
        fprintf(stderr,"I lost my shadow child report (exit %d, signal %d, bytes %ld).\n",
            WIFEXITED(status)?WEXITSTATUS(status):-1,WIFSIGNALED(status)?WTERMSIG(status):0,(long)got);return report;
    }
    if (received.completed > count ||
        (received.status == NL_SERVICE_SHADOW_OK && received.completed != count)) return report;
    return received;
}

NlServiceShadowReport nl_service_run_shadows(const NlServiceShadow *suite,size_t count,
    bool allowed,const char *path) {
    return nl_service_run_catalog_shadows(suite,count,1,allowed,path);
}
