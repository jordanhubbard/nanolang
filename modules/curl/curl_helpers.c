/**
 * curl_helpers.c - Simplified curl wrapper functions for nanolang
 * 
 * Provides easy-to-use HTTP request functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>

/* Memory buffer for storing response data */
typedef struct {
    char *data;
    size_t size;
} MemoryBuffer;

/* Callback function for writing response data */
static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    MemoryBuffer *mem = (MemoryBuffer *)userp;
    
    char *ptr = realloc(mem->data, mem->size + realsize + 1);
    if (ptr == NULL) {
        fprintf(stderr, "curl_helpers: Out of memory\n");
        return 0;
    }
    
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0; /* null terminate */
    
    return realsize;
}

/**
 * Initialize curl globally (call once at program start)
 * Returns 0 on success, non-zero on failure
 */
int64_t nl_curl_global_init(void) {
    return (int64_t)curl_global_init(CURL_GLOBAL_DEFAULT);
}

/**
 * Cleanup curl globally (call once at program end)
 */
void nl_curl_global_cleanup(void) {
    curl_global_cleanup();
}

/**
 * Create a new curl handle
 * Returns handle ID (pointer as int64), or 0 on failure
 */
int64_t nl_curl_easy_init(void) {
    CURL *curl = curl_easy_init();
    return (int64_t)curl;
}

/**
 * Cleanup a curl handle
 */
void nl_curl_easy_cleanup(int64_t handle) {
    CURL *curl = (CURL *)handle;
    if (curl) {
        curl_easy_cleanup(curl);
    }
}

/**
 * Set URL for request
 * Returns 0 on success
 */
int64_t nl_curl_easy_setopt_url(int64_t handle, const char *url) {
    CURL *curl = (CURL *)handle;
    if (!curl) return 1;
    return (int64_t)curl_easy_setopt(curl, CURLOPT_URL, url);
}

/**
 * Enable/disable following redirects
 * Returns 0 on success
 */
int64_t nl_curl_easy_setopt_follow_location(int64_t handle, int64_t follow) {
    CURL *curl = (CURL *)handle;
    if (!curl) return 1;
    return (int64_t)curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, (long)follow);
}

/**
 * Set timeout in seconds
 * Returns 0 on success
 */
int64_t nl_curl_easy_setopt_timeout(int64_t handle, int64_t timeout_secs) {
    CURL *curl = (CURL *)handle;
    if (!curl) return 1;
    return (int64_t)curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)timeout_secs);
}

/**
 * Set custom User-Agent header
 * Returns 0 on success
 */
int64_t nl_curl_easy_setopt_useragent(int64_t handle, const char *useragent) {
    CURL *curl = (CURL *)handle;
    if (!curl) return 1;
    return (int64_t)curl_easy_setopt(curl, CURLOPT_USERAGENT, useragent);
}

/**
 * Perform HTTP GET request and return response body
 * Returns response as string, or empty string on failure
 * NOTE: Caller should free the returned string when done
 */
const char* nl_curl_simple_get(const char *url) {
    CURL *curl;
    CURLcode res;
    MemoryBuffer chunk = {0};
    
    chunk.data = malloc(1);
    chunk.size = 0;
    
    curl = curl_easy_init();
    if (!curl) {
        free(chunk.data);
        return strdup("");
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "nanolang-curl/1.0");
    
    res = curl_easy_perform(curl);
    
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "curl_simple_get failed: %s\n", curl_easy_strerror(res));
        free(chunk.data);
        return strdup("");
    }
    
    return chunk.data ? chunk.data : strdup("");
}

/**
 * Perform HTTP POST request with data
 * Returns response as string, or empty string on failure
 */
const char* nl_curl_simple_post(const char *url, const char *data) {
    CURL *curl;
    CURLcode res;
    MemoryBuffer chunk = {0};
    
    chunk.data = malloc(1);
    chunk.size = 0;
    
    curl = curl_easy_init();
    if (!curl) {
        free(chunk.data);
        return strdup("");
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "nanolang-curl/1.0");
    
    res = curl_easy_perform(curl);
    
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "curl_simple_post failed: %s\n", curl_easy_strerror(res));
        free(chunk.data);
        return strdup("");
    }
    
    return chunk.data ? chunk.data : strdup("");
}

/**
 * Get HTTP response code from last request
 * Returns HTTP code (200, 404, etc.) or 0 on failure
 */
int64_t nl_curl_easy_getinfo_response_code(int64_t handle) {
    CURL *curl = (CURL *)handle;
    long response_code = 0;
    
    if (!curl) return 0;
    
    CURLcode res = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    if (res != CURLE_OK) {
        return 0;
    }
    
    return (int64_t)response_code;
}

/**
 * Perform the request (for advanced usage with manual setup)
 * Returns 0 on success
 */
int64_t nl_curl_easy_perform(int64_t handle) {
    CURL *curl = (CURL *)handle;
    if (!curl) return 1;
    return (int64_t)curl_easy_perform(curl);
}

/**
 * Download a file from URL to local path
 * Returns 0 on success, non-zero on failure
 */
int64_t nl_curl_download_file(const char *url, const char *output_path) {
    CURL *curl;
    FILE *fp;
    CURLcode res;
    
    curl = curl_easy_init();
    if (!curl) {
        return 1;
    }
    
    fp = fopen(output_path, "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        return 2;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L); /* 5 min timeout for downloads */
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "nanolang-curl/1.0");
    
    res = curl_easy_perform(curl);
    
    fclose(fp);
    curl_easy_cleanup(curl);
    
    return (res == CURLE_OK) ? 0 : (int64_t)res;
}
