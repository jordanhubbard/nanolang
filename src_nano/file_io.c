/* File I/O implementation for nanolang compiler */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Read entire file into a string */
const char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file '%s'\n", path);
        return "";
    }
    
    /* Get file size */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    /* Allocate buffer */
    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fclose(f);
        fprintf(stderr, "Error: Could not allocate memory for file '%s'\n", path);
        return "";
    }
    
    /* Read file */
    size_t bytes_read = fread(buffer, 1, size, f);
    buffer[bytes_read] = '\0';
    
    fclose(f);
    return buffer;
}

/* Write string to file */
long long write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not open file '%s' for writing\n", path);
        return 1;
    }
    
    size_t len = strlen(content);
    size_t written = fwrite(content, 1, len, f);
    
    fclose(f);
    
    if (written != len) {
        fprintf(stderr, "Error: Could not write all data to file '%s'\n", path);
        return 1;
    }
    
    return 0;
}

/* Check if file exists */
int file_exists(const char* path) {
    struct stat buffer;
    return (stat(path, &buffer) == 0) ? 1 : 0;
}
