#include "nano_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int64_t eval(const char *source) {
    if (!source) {
        return 1;
    }

    char src_template[] = "/tmp/nano_eval_XXXXXX";
    int src_fd = mkstemp(src_template);
    if (src_fd < 0) {
        return 1;
    }

    FILE *src_file = fdopen(src_fd, "w");
    if (!src_file) {
        close(src_fd);
        unlink(src_template);
        return 1;
    }

    fputs(source, src_file);
    fclose(src_file);

    char bin_template[] = "/tmp/nano_eval_bin_XXXXXX";
    int bin_fd = mkstemp(bin_template);
    if (bin_fd < 0) {
        unlink(src_template);
        return 1;
    }
    close(bin_fd);

    char compile_cmd[2048];
    snprintf(compile_cmd, sizeof(compile_cmd), "./bin/nanoc %s -o %s", src_template, bin_template);
    int compile_status = system(compile_cmd);

    if (compile_status != 0) {
        unlink(src_template);
        unlink(bin_template);
        return 1;
    }

    int run_status = system(bin_template);

    unlink(src_template);
    unlink(bin_template);

    if (WIFEXITED(run_status)) {
        return WEXITSTATUS(run_status) == 0 ? 0 : 1;
    }

    return 1;
}
