#ifndef NANO_HOST_ARTIFACT_LOADER_H
#define NANO_HOST_ARTIFACT_LOADER_H
/* I verify content before loading. This deployment check is not a sandbox
 * against a writer racing replacement of executable or library files. */
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
static inline void *nano_host_load(const char *record) {
    static const char prefix[]="NANOLANG_HOST_V1:";
    if (strncmp(record,prefix,sizeof prefix-1)) return NULL;
    const char *digest=record+sizeof prefix-1;
    if (strlen(digest)<66 || digest[64]!=':') return NULL;
    for (unsigned i=0;i<64;++i) if (!((digest[i]>='0'&&digest[i]<='9')||(digest[i]>='a'&&digest[i]<='f'))) return NULL;
    const char *original=digest+65;
    if (original[0]!='/' || strlen(original)>=PATH_MAX) return NULL;
    char executable[PATH_MAX],resolved[PATH_MAX],packaged[PATH_MAX];
#ifdef __APPLE__
    uint32_t size=sizeof executable;
    if (_NSGetExecutablePath(executable,&size)) return NULL;
#else
    ssize_t size=readlink("/proc/self/exe",executable,sizeof executable-1);
    if (size<=0 || (size_t)size>=sizeof executable-1) return NULL;
    executable[size]=0;
#endif
    if (!realpath(executable,resolved)) return NULL;
    char *slash=strrchr(resolved,'/');if (!slash) return NULL;*slash=0;
    int n=snprintf(packaged,sizeof packaged,"%s/../lib/nanolang/hosts",resolved);
    if (n<0 || (size_t)n>=sizeof packaged) return NULL;
    struct stat status;const char *path=original;
    if (!stat(packaged,&status)) {
        if (!S_ISDIR(status.st_mode)) return NULL;
        size_t used=strlen(packaged);
        n=snprintf(packaged+used,sizeof packaged-used,"/%.64s",digest);
        if (n<0 || (size_t)n>=sizeof packaged-used) return NULL;
        path=packaged;
    } else if (errno!=ENOENT) return NULL;
    if (stat(path,&status) || !S_ISREG(status.st_mode)) return NULL;
    char actual[65];
    if (!nano_host_digest_file(path,actual) || memcmp(actual,digest,64)) return NULL;
    return dlopen(path,RTLD_NOW|RTLD_LOCAL);
}
#endif
