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
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/random.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <linux/fs.h>
#endif

#define FP_PATH 4096u
#define FP_NAME 256u
#define FP_INTERRUPTS 64u
#define FP_ATTEMPTS 64u

typedef struct {
    const char *name;
    int fd;
    bool created;
    bool identified;
    struct stat identity;
} FpChild;

typedef struct {
    NlFileBindingPublishReport *report;
    int parent;
    int stage;
    bool stage_created;
    bool stage_identified;
    struct stat parent_identity;
    struct stat stage_identity;
    FpChild child[2];
} FpTransaction;

static void fp_fail(FpTransaction *t, NlFileBindingStatus status,
                    NlFilePublishStage stage, int error) {
    if (t->report->status == NL_FILE_BINDING_OK) {
        t->report->status = status;
        t->report->failed_stage = stage;
        t->report->first_errno = error;
    }
}
static void fp_cleanup_error(FpTransaction *t, int error) {
    if (!t->report->cleanup_errno) t->report->cleanup_errno = error ? error : EIO;
    t->report->cleanup_pending = true;
}
/* I invalidate my descriptor slot before the one close attempt. */
static bool fp_close(FpTransaction *t, int *slot, bool cleanup) {
    if (*slot < 0) return true;
    int fd = *slot;
    *slot = -1;
    if (close(fd) == 0) return true;
    int error = errno;
    if (cleanup || t->report->status != NL_FILE_BINDING_OK)
        fp_cleanup_error(t, error);
    else {
        fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_CLOSE, error);
        t->report->cleanup_pending = true;
    }
    return false;
}
static bool fp_same(const struct stat *a, const struct stat *b) {
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
           (a->st_mode & S_IFMT) == (b->st_mode & S_IFMT);
}
static bool fp_sync(FpTransaction *t, int fd, NlFilePublishStage stage) {
    unsigned interrupted = 0;
    for (;;) {
        if (fsync(fd) == 0) return true;
        int error = errno;
        if (error == EINTR && interrupted++ < FP_INTERRUPTS) continue;
        fp_fail(t, NL_FILE_BINDING_IO, stage, error);
        return false;
    }
}
static bool fp_write(FpTransaction *t, int fd, const unsigned char *bytes,
                     size_t size, NlFilePublishStage stage) {
    size_t written = 0;
    unsigned interrupted = 0;
    while (written < size) {
        ssize_t count = write(fd, bytes + written, size - written);
        if (count > 0) {
            if ((size_t)count > size - written) {
                fp_fail(t, NL_FILE_BINDING_IO, stage, EIO);
                return false;
            }
            written += (size_t)count;
        } else {
            int error = count == 0 ? EIO : errno;
            if (error == EINTR && interrupted++ < FP_INTERRUPTS) continue;
            fp_fail(t, NL_FILE_BINDING_IO, stage, error);
            return false;
        }
    }
    return true;
}
static NlFileBindingStatus fp_path(const char *path, char parent[FP_PATH],
                                   char name[FP_NAME]) {
    size_t size = strnlen(path, FP_PATH);
    if (size == FP_PATH) return NL_FILE_BINDING_LIMIT;
    if (!size || path[size - 1] == '/') return NL_FILE_BINDING_INVALID;
    const char *slash = strrchr(path, '/');
    const char *last = slash ? slash + 1 : path;
    size_t count = size - (size_t)(last - path);
    if (count >= FP_NAME) return NL_FILE_BINDING_LIMIT;
    if (!count || (count == 1 && last[0] == '.') ||
        (count == 2 && last[0] == '.' && last[1] == '.'))
        return NL_FILE_BINDING_INVALID;
    memcpy(name, last, count);
    name[count] = 0;
    if (!slash) { parent[0] = '.'; parent[1] = 0; }
    else {
        size_t prefix = slash == path ? 1 : (size_t)(slash - path);
        while (prefix > 1 && path[prefix - 1] == '/') prefix--;
        memcpy(parent, path, prefix);
        parent[prefix] = 0;
    }
    return NL_FILE_BINDING_OK;
}
static bool fp_stage(FpTransaction *t) {
    static const char hex[] = "0123456789abcdef";
    static const char prefix[] = ".nsi-file-binding-";
    for (unsigned attempt = 0; attempt < FP_ATTEMPTS; attempt++) {
        unsigned char random[16];
        if (getentropy(random, sizeof random) != 0) {
            fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_STAGE, errno);
            return false;
        }
        char candidate[96];
        memcpy(candidate, prefix, sizeof prefix - 1);
        for (size_t i = 0; i < sizeof random; i++) {
            candidate[sizeof prefix - 1 + 2*i] = hex[random[i] >> 4];
            candidate[sizeof prefix + 2*i] = hex[random[i] & 15];
        }
        candidate[sizeof prefix - 1 + 2*sizeof random] = 0;
        if (mkdirat(t->parent, candidate, 0700) != 0) {
            int error = errno;
            if (error == EEXIST) continue;
            fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_STAGE, error);
            return false;
        }
        t->stage_created = true;
        memcpy(t->report->staging_name, candidate, strlen(candidate) + 1);
        t->stage = openat(t->parent, candidate,
                         O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (t->stage < 0) {
            fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_STAGE, errno);
            return false;
        }
        if (fstat(t->stage, &t->stage_identity) != 0) {
            fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_STAGE, errno);
            return false;
        }
        if (!S_ISDIR(t->stage_identity.st_mode)) {
            fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_STAGE, ESTALE);
            return false;
        }
        t->stage_identified = true;
        struct stat entry;
        if (fstatat(t->parent, candidate, &entry, AT_SYMLINK_NOFOLLOW) != 0) {
            fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_STAGE, errno);
            return false;
        }
        if (!fp_same(&entry, &t->stage_identity)) {
            fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_STAGE, ESTALE);
            return false;
        }
        return true;
    }
    fp_fail(t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_STAGE, EEXIST);
    return false;
}
static bool fp_file(FpTransaction *t, size_t index, const unsigned char *bytes,
                    size_t size, NlFilePublishStage phase) {
    FpChild *c = &t->child[index];
    c->fd = openat(t->stage, c->name,
                  O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (c->fd < 0) { fp_fail(t, NL_FILE_BINDING_IO, phase, errno); return false; }
    c->created = true;
    if (fstat(c->fd, &c->identity) != 0) {
        fp_fail(t, NL_FILE_BINDING_IO, phase, errno);
        return false;
    }
    if (!S_ISREG(c->identity.st_mode)) {
        fp_fail(t, NL_FILE_BINDING_IO, phase, ESTALE);
        return false;
    }
    c->identified = true;
    if (!fp_write(t, c->fd, bytes, size, phase) || !fp_sync(t, c->fd, phase))
        return false;
    return fp_close(t, &c->fd, false);
}
static void fp_remove_child(FpTransaction *t, FpChild *c) {
    if (!c->created) return;
    if (t->stage < 0) { fp_cleanup_error(t, ESTALE); return; }
    struct stat actual;
    if (fstatat(t->stage, c->name, &actual, AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno != ENOENT) fp_cleanup_error(t, errno);
        return;
    }
    if (!c->identified || !fp_same(&actual, &c->identity)) {
        fp_cleanup_error(t, ESTALE);
        return;
    }
    if (unlinkat(t->stage, c->name, 0) != 0 && errno != ENOENT)
        fp_cleanup_error(t, errno);
}
static void fp_rollback(FpTransaction *t) {
    for (size_t i = 0; i < 2; i++) {
        fp_close(t, &t->child[i].fd, true);
        fp_remove_child(t, &t->child[i]);
    }
    if (!t->stage_created) return;
    struct stat actual;
    if (fstatat(t->parent, t->report->staging_name, &actual,
                AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno != ENOENT) fp_cleanup_error(t, errno);
        return;
    }
    if (!t->stage_identified || !fp_same(&actual, &t->stage_identity)) {
        fp_cleanup_error(t, ESTALE);
        return;
    }
    if (unlinkat(t->parent, t->report->staging_name, AT_REMOVEDIR) != 0 &&
        errno != ENOENT) fp_cleanup_error(t, errno);
}
static int fp_rename(int parent, const char *stage, const char *final_name) {
#if defined(__linux__) && defined(SYS_renameat2)
    return (int)syscall(SYS_renameat2, parent, stage, parent, final_name, RENAME_NOREPLACE);
#elif defined(__APPLE__) && defined(RENAME_EXCL)
    return renameatx_np(parent, stage, parent, final_name, RENAME_EXCL);
#else
    (void)parent; (void)stage; (void)final_name;
    errno = ENOSYS;
    return -1;
#endif
}
NlFileBindingStatus nl_file_binding_publish(const NlFileBindingPlan *plan,
    const char *directory, NlFileBindingPublishReport *report) {
    if (!report) return NL_FILE_BINDING_INVALID;
    memset(report, 0, sizeof *report);
    FpTransaction t;
    memset(&t, 0, sizeof t);
    t.report = report; t.parent = -1; t.stage = -1;
    t.child[0].name = "interface.nsi.json"; t.child[0].fd = -1;
    t.child[1].name = "binding.nano"; t.child[1].fd = -1;
    if (!plan || !directory) {
        fp_fail(&t, NL_FILE_BINDING_INVALID, NL_FILE_PUBLISH_VALIDATE, 0);
        return report->status;
    }
    char parent[FP_PATH], final_name[FP_NAME];
    NlFileBindingStatus status = fp_path(directory, parent, final_name);
    size_t lengths[2] = {0, 0};
    const unsigned char *bytes[2] = {
        nl_file_binding_interface_bytes(plan, &lengths[0]),
        nl_file_binding_source_bytes(plan, &lengths[1])
    };
    if (status == NL_FILE_BINDING_OK &&
        (!bytes[0] || !bytes[1] || !lengths[0] || !lengths[1]))
        status = NL_FILE_BINDING_INVALID;
    if (status == NL_FILE_BINDING_OK &&
        (lengths[0] > NL_FILE_BINDING_MAX_BYTES || lengths[1] > NL_FILE_BINDING_MAX_BYTES))
        status = NL_FILE_BINDING_LIMIT;
    if (status != NL_FILE_BINDING_OK) {
        fp_fail(&t, status, NL_FILE_PUBLISH_VALIDATE, 0);
        return report->status;
    }
#if !(defined(__linux__) && defined(SYS_renameat2)) && !(defined(__APPLE__) && defined(RENAME_EXCL))
    fp_fail(&t, NL_FILE_BINDING_UNSUPPORTED, NL_FILE_PUBLISH_VALIDATE, 0);
    return report->status;
#endif
    t.parent = open(parent, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (t.parent < 0) {
        fp_fail(&t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_PARENT, errno);
        return report->status;
    }
    if (fstat(t.parent, &t.parent_identity) != 0) {
        fp_fail(&t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_PARENT, errno);
        goto finish;
    }
    if (!S_ISDIR(t.parent_identity.st_mode)) {
        fp_fail(&t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_PARENT, ENOTDIR);
        goto finish;
    }
    if (!fp_stage(&t) ||
        !fp_file(&t, 0, bytes[0], lengths[0], NL_FILE_PUBLISH_INTERFACE) ||
        !fp_file(&t, 1, bytes[1], lengths[1], NL_FILE_PUBLISH_SOURCE) ||
        !fp_sync(&t, t.stage, NL_FILE_PUBLISH_STAGE_SYNC)) goto finish;
    struct stat actual;
    if (fstatat(t.parent, report->staging_name, &actual, AT_SYMLINK_NOFOLLOW) != 0) {
        fp_fail(&t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_RENAME, errno);
        goto finish;
    }
    if (!fp_same(&actual, &t.stage_identity)) {
        fp_fail(&t, NL_FILE_BINDING_IO, NL_FILE_PUBLISH_RENAME, ESTALE);
        goto finish;
    }
    if (fp_rename(t.parent, report->staging_name, final_name) != 0) {
        int error = errno;
        status = NL_FILE_BINDING_IO;
        if (error == EEXIST || error == ENOTEMPTY) status = NL_FILE_BINDING_EXISTS;
        else if (error == ENOSYS || error == ENOTSUP || error == EOPNOTSUPP || error == EINVAL)
            status = NL_FILE_BINDING_UNSUPPORTED;
        fp_fail(&t, status, NL_FILE_PUBLISH_RENAME, error);
        goto finish;
    }
    report->published = true;
    report->staging_name[0] = 0;
    if (fp_sync(&t, t.parent, NL_FILE_PUBLISH_PARENT_SYNC)) report->durable = true;
finish:
    if (!report->published) fp_rollback(&t);
    fp_close(&t, &t.stage, report->status != NL_FILE_BINDING_OK);
    fp_close(&t, &t.parent, report->status != NL_FILE_BINDING_OK);
    return report->status;
}
