#include "../src/shell_path.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

int main(void) {
    const char *words[] = {
        "space path", "single'quote", "double\"quote", "dollar$(touch sentinel)",
        "back\\slash", "line\nbreak", "; touch sentinel"
    };
    char command[4096] = "set --";
    size_t pos = strlen(command);
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++) {
        assert(shell_append_word(command, sizeof(command), &pos, words[i]));
    }
    assert(shell_append_text(command, sizeof(command), &pos,
        "; test \"$#\" -eq 7 && test \"$1\" = 'space path' && "
        "test \"$2\" = \"single'quote\" && test \"$3\" = 'double\"quote' && "
        "test \"$4\" = 'dollar$(touch sentinel)' && test \"$5\" = 'back\\slash' && "
        "test \"$6\" = 'line\nbreak' && test \"$7\" = '; touch sentinel'"));
    int status = system(command);
    assert(status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(remove("sentinel") != 0);

    char small[4] = {0};
    pos = 0;
    assert(!shell_append_word(small, sizeof(small), &pos, "too long"));
    puts("module builder path quoting passed");
    return 0;
}
