/* I compare descriptor, errno, mode, fork and output behavior with observation enabled. */
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#define REQUIRE(value) do { if (!(value)) return 1; } while (0)
int main(int argc, char **argv) {
    REQUIRE(argc == 2);
    errno = 0; REQUIRE(dup(-1) == -1 && errno == EBADF);
    errno = 0; REQUIRE(dup2(-1, STDOUT_FILENO) == -1 && errno == EBADF);
    errno = 0; REQUIRE(close(-1) == -1 && errno == EBADF);
    errno = E2BIG;
    int null = open("/dev/null", O_WRONLY);
    REQUIRE(null >= 0 && errno == E2BIG);
    REQUIRE(close(null) == 0);
    int file = open(argv[1], O_CREAT | O_EXCL | O_WRONLY, 0600);
    REQUIRE(file >= 0 && write(file, "ok\n", 3) == 3 && close(file) == 0);
    struct stat info;
    REQUIRE(stat(argv[1], &info) == 0 && (info.st_mode & 0777) == 0600);
    file = open(argv[1], O_RDONLY);
    char contents[3];
    REQUIRE(file >= 0 && read(file, contents, 3) == 3 && memcmp(contents, "ok\n", 3) == 0);
    REQUIRE(close(file) == 0 && unlink(argv[1]) == 0);
    int channel[2];
    REQUIRE(pipe(channel) == 0);
    pid_t child = fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        if (close(channel[0]) || write(channel[1], "ok\n", 3) != 3 || close(channel[1])) _exit(8);
        _exit(7);
    }
    REQUIRE(close(channel[1]) == 0 && read(channel[0], contents, 3) == 3);
    REQUIRE(close(channel[0]) == 0);
    int status;
    REQUIRE(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 7);
    REQUIRE(memcmp(contents, "ok\n", 3) == 0);
    puts("I preserved descriptor, errno, creation mode, data, fork and exit behavior.");
    return 0;
}
