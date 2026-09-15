#include "../src/shell_path.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    char directory[] = "/tmp/nano-shell-path-XXXXXX";
    char *previous = getcwd(NULL, 0);
    assert(previous && mkdtemp(directory) && chdir(directory) == 0);
    const char *words[] = {
        "space path", "single'quote", "double\"quote", "dollar$(touch sentinel)",
        "back\\slash", "line\nbreak", "; touch sentinel"
    };
    char command[4096] = "set --";
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++) {
        assert(module_append_path_flag(command, sizeof(command), "", words[i]));
    }
    const char *check =
        "; test \"$#\" -eq 7 && test \"$1\" = 'space path' && "
        "test \"$2\" = \"single'quote\" && test \"$3\" = 'double\"quote' && "
        "test \"$4\" = 'dollar$(touch sentinel)' && test \"$5\" = 'back\\slash' && "
        "test \"$6\" = 'line\nbreak' && test \"$7\" = '; touch sentinel'";
    assert(strlen(command) + strlen(check) < sizeof(command));
    strcat(command, check);
    int status = system(command);
    assert(status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(access("sentinel", F_OK) != 0);

    char small[4] = {0};
    assert(!module_append_path_flag(small, sizeof(small), "", "too long"));
    assert(chdir(previous) == 0);
    free(previous);
    assert(rmdir(directory) == 0);
    puts("I preserved literal module paths through shell quoting.");
    return 0;
}
