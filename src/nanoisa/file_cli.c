#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "file_cli.h"
#include "file_hosted.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void file_cli_error(char *out, size_t size, const char *operation, int error) {
    if (out && size)
        (void)snprintf(out, size, "I cannot %s: %s", operation, strerror(error ? error : EIO));
}
bool nvm_file_cli_read(const char *path, uint8_t **out, size_t *out_size,
                       char *error, size_t error_size) {
    if (!path || !out || !out_size || (error_size && !error)) return false;
    FILE *input = fopen(path, "rb");
    if (!input) { file_cli_error(error,error_size,"open File input",errno); return false; }
    uint8_t *bytes = NULL;
    size_t size = 0;
    int saved = 0;
    if (fseek(input,0,SEEK_END)) saved = errno ? errno : EIO;
    long extent = saved ? -1 : ftell(input);
    if (!saved && extent < 0) saved = errno ? errno : EIO;
    if (!saved && (extent == 0 || (uint64_t)extent > NVM_FILE_HOSTED_INPUT_BYTES)) saved = EFBIG;
    if (!saved && fseek(input,0,SEEK_SET)) saved = errno ? errno : EIO;
    if (!saved) {
        size = (size_t)extent;
        bytes = malloc(size);
        if (!bytes) saved = ENOMEM;
    }
    if (!saved && fread(bytes,1,size,input) != size) saved = errno ? errno : EIO;
    if (!saved && (fgetc(input) != EOF || ferror(input))) saved = EIO;
    if (fclose(input) && !saved) saved = errno ? errno : EIO;
    if (saved) {
        free(bytes);
        file_cli_error(error,error_size,"read bounded File input",saved);
        return false;
    }
    *out = bytes;
    *out_size = size;
    if (error_size) error[0] = '\0';
    return true;
}

bool nvm_file_cli_write(const char *path, const char *text, char *error, size_t error_size) {
    if (!path || !text || (error_size && !error)) return false;
    static const char suffix[] = ".nano-file-XXXXXX";
    size_t length = strlen(path);
    if (length > SIZE_MAX - sizeof suffix) {
        file_cli_error(error,error_size,"stage File output",EOVERFLOW);
        return false;
    }
    char *temporary = malloc(length + sizeof suffix);
    if (!temporary) { file_cli_error(error,error_size,"stage File output",ENOMEM); return false; }
    memcpy(temporary,path,length);
    memcpy(temporary+length,suffix,sizeof suffix);
    int fd = mkstemp(temporary), saved = 0;
    if (fd < 0) saved = errno ? errno : EIO;
    else {
        FILE *output = fdopen(fd,"wb");
        if (!output) {
            saved = errno ? errno : EIO;
            (void)close(fd);
        } else {
            size_t bytes = strlen(text);
            if (fwrite(text,1,bytes,output) != bytes) saved = errno ? errno : EIO;
            if (fflush(output) && !saved) saved = errno ? errno : EIO;
            if (fclose(output) && !saved) saved = errno ? errno : EIO;
        }
        if (!saved && rename(temporary,path)) saved = errno ? errno : EIO;
        if (saved) (void)unlink(temporary);
    }
    free(temporary);
    if (saved) { file_cli_error(error,error_size,"publish File C output",saved); return false; }
    if (error_size) error[0] = '\0';
    return true;
}
