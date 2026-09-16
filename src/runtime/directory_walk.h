#ifndef NANOLANG_DIRECTORY_WALK_H
#define NANOLANG_DIRECTORY_WALK_H

#include "dyn_array.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

typedef struct {
    char *path;
    dev_t device;
    ino_t inode;
    bool visited;
} NlWalkDirectory;

/* I retain queue entries as the visited-identity set, but release path bytes
 * after visiting. Reallocation never invalidates the separately allocated path. */
static void nl_walk_enqueue(NlWalkDirectory **queue, size_t *count, size_t *capacity, char *path) {
    if (*count == *capacity) {
        if (*capacity > SIZE_MAX / 2 / sizeof(**queue)) abort();
        size_t next = *capacity ? *capacity * 2 : 16;
        NlWalkDirectory *grown = realloc(*queue, next * sizeof(**queue));
        if (!grown) abort();
        *queue = grown;
        *capacity = next;
    }
    (*queue)[(*count)++] = (NlWalkDirectory){.path = path};
}

/* I follow links, but visit each opened directory identity once. I keep one
 * DIR open at a time and do not consume C stack in proportion to tree depth. */
static DynArray* nl_fs_walkdir(const char* root) {
    DynArray* result = dyn_array_new_with_capacity(ELEM_STRING, 128);
    if (!result || !root || !root[0]) return result;
    NlWalkDirectory *queue = NULL;
    size_t count = 0, capacity = 0;
    char *initial = strdup(root);
    if (!initial) abort();
    nl_walk_enqueue(&queue, &count, &capacity, initial);
    for (size_t current = 0; current < count; ++current) {
        char *path = queue[current].path;
        DIR *dir = opendir(path);
        struct stat identity;
        bool seen = false;
        if (dir && fstat(dirfd(dir), &identity) == 0) {
            for (size_t prior = 0; prior < current; ++prior) {
                if (queue[prior].visited && queue[prior].device == identity.st_dev &&
                    queue[prior].inode == identity.st_ino) { seen = true; break; }
            }
            if (!seen) {
                queue[current].visited = true;
                queue[current].device = identity.st_dev;
                queue[current].inode = identity.st_ino;
                struct dirent *entry;
                while ((entry = readdir(dir)) != NULL) {
                    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
                    size_t parent_len = strlen(path), name_len = strlen(entry->d_name);
                    if (parent_len > SIZE_MAX - name_len - 2) abort();
                    char *child = malloc(parent_len + name_len + 2);
                    if (!child) abort();
                    memcpy(child, path, parent_len);
                    child[parent_len] = '/';
                    memcpy(child + parent_len + 1, entry->d_name, name_len + 1);
                    struct stat st;
                    if (stat(child, &st) == 0) {
                        if (S_ISREG(st.st_mode)) dyn_array_push_string_copy(result, child);
                        else if (S_ISDIR(st.st_mode)) {
                            nl_walk_enqueue(&queue, &count, &capacity, child);
                            child = NULL;
                        }
                    }
                    free(child);
                }
            }
        }
        if (dir) closedir(dir);
        free(path);
        queue[current].path = NULL;
    }
    free(queue);
    return result;
}

#endif
