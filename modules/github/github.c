#include "github.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <curl/curl.h>

#define GITHUB_API_BASE "https://api.github.com"
#define MAX_URL_LEN 1024
#define MAX_HEADER_LEN 512
#define MAX_JSON_LEN 16384

// Memory structure for curl response
struct MemoryStruct {
    char *memory;
    size_t size;
};

// Callback for curl writes
static size_t write_memory_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;

    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (!ptr) {
        return 0;  // Out of memory
    }

    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;

    return realsize;
}

// Perform HTTP GET request with authentication
static char* github_get(const char* url, const char* token) {
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    chunk.memory = malloc(1);
    chunk.size = 0;

    curl = curl_easy_init();
    if (!curl) {
        free(chunk.memory);
        return strdup("{\"error\": \"Failed to initialize curl\"}");
    }

    // Build authorization header
    char auth_header[MAX_HEADER_LEN];
    snprintf(auth_header, sizeof(auth_header), "Authorization: token %s", token);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, auth_header);
    headers = curl_slist_append(headers, "User-Agent: nanolang-github-client");
    headers = curl_slist_append(headers, "Accept: application/vnd.github.v3+json");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_memory_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "{\"error\": \"curl failed: %s\"}", curl_easy_strerror(res));
        free(chunk.memory);
        chunk.memory = strdup(error_msg);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return chunk.memory;
}

// Perform HTTP POST request with authentication
static char* github_post(const char* url, const char* json_body, const char* token) {
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    chunk.memory = malloc(1);
    chunk.size = 0;

    curl = curl_easy_init();
    if (!curl) {
        free(chunk.memory);
        return strdup("{\"error\": \"Failed to initialize curl\"}");
    }

    // Build authorization header
    char auth_header[MAX_HEADER_LEN];
    snprintf(auth_header, sizeof(auth_header), "Authorization: token %s", token);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, auth_header);
    headers = curl_slist_append(headers, "User-Agent: nanolang-github-client");
    headers = curl_slist_append(headers, "Accept: application/vnd.github.v3+json");
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_memory_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "{\"error\": \"curl failed: %s\"}", curl_easy_strerror(res));
        free(chunk.memory);
        chunk.memory = strdup(error_msg);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return chunk.memory;
}

// Perform HTTP PATCH request with authentication
static char* github_patch(const char* url, const char* json_body, const char* token) {
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    chunk.memory = malloc(1);
    chunk.size = 0;

    curl = curl_easy_init();
    if (!curl) {
        free(chunk.memory);
        return strdup("{\"error\": \"Failed to initialize curl\"}");
    }

    // Build authorization header
    char auth_header[MAX_HEADER_LEN];
    snprintf(auth_header, sizeof(auth_header), "Authorization: token %s", token);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, auth_header);
    headers = curl_slist_append(headers, "User-Agent: nanolang-github-client");
    headers = curl_slist_append(headers, "Accept: application/vnd.github.v3+json");
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PATCH");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_memory_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "{\"error\": \"curl failed: %s\"}", curl_easy_strerror(res));
        free(chunk.memory);
        chunk.memory = strdup(error_msg);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return chunk.memory;
}

// Get repository information
char* nl_github_get_repo(const char* owner, const char* repo, const char* token) {
    char url[MAX_URL_LEN];
    snprintf(url, sizeof(url), "%s/repos/%s/%s", GITHUB_API_BASE, owner, repo);
    return github_get(url, token);
}

// List issues (state can be "open", "closed", or "all")
char* nl_github_list_issues(const char* owner, const char* repo, const char* state, const char* token) {
    char url[MAX_URL_LEN];
    snprintf(url, sizeof(url), "%s/repos/%s/%s/issues?state=%s", GITHUB_API_BASE, owner, repo, state);
    return github_get(url, token);
}

// Get specific issue
char* nl_github_get_issue(const char* owner, const char* repo, int64_t issue_number, const char* token) {
    char url[MAX_URL_LEN];
    snprintf(url, sizeof(url), "%s/repos/%s/%s/issues/%lld", GITHUB_API_BASE, owner, repo, (long long)issue_number);
    return github_get(url, token);
}

// Create new issue
char* nl_github_create_issue(const char* owner, const char* repo, const char* title, const char* body, const char* token) {
    char url[MAX_URL_LEN];
    char json[MAX_JSON_LEN];

    snprintf(url, sizeof(url), "%s/repos/%s/%s/issues", GITHUB_API_BASE, owner, repo);

    // Build JSON body (simple escaping - replace " with \")
    snprintf(json, sizeof(json), "{\"title\":\"%s\",\"body\":\"%s\"}", title, body);

    return github_post(url, json, token);
}

// Update existing issue
char* nl_github_update_issue(const char* owner, const char* repo, int64_t issue_number,
                              const char* title, const char* body, const char* state, const char* token) {
    char url[MAX_URL_LEN];
    char json[MAX_JSON_LEN];

    snprintf(url, sizeof(url), "%s/repos/%s/%s/issues/%lld", GITHUB_API_BASE, owner, repo, (long long)issue_number);

    // Build JSON body with all fields
    if (title && body && state) {
        snprintf(json, sizeof(json), "{\"title\":\"%s\",\"body\":\"%s\",\"state\":\"%s\"}", title, body, state);
    } else if (title && body) {
        snprintf(json, sizeof(json), "{\"title\":\"%s\",\"body\":\"%s\"}", title, body);
    } else if (state) {
        snprintf(json, sizeof(json), "{\"state\":\"%s\"}", state);
    } else {
        return strdup("{\"error\": \"At least one field (title, body, or state) must be provided\"}");
    }

    return github_patch(url, json, token);
}

// Create issue comment
char* nl_github_create_issue_comment(const char* owner, const char* repo, int64_t issue_number,
                                      const char* body, const char* token) {
    char url[MAX_URL_LEN];
    char json[MAX_JSON_LEN];

    snprintf(url, sizeof(url), "%s/repos/%s/%s/issues/%lld/comments", GITHUB_API_BASE, owner, repo, (long long)issue_number);
    snprintf(json, sizeof(json), "{\"body\":\"%s\"}", body);

    return github_post(url, json, token);
}

// List pull requests (state can be "open", "closed", or "all")
char* nl_github_list_prs(const char* owner, const char* repo, const char* state, const char* token) {
    char url[MAX_URL_LEN];
    snprintf(url, sizeof(url), "%s/repos/%s/%s/pulls?state=%s", GITHUB_API_BASE, owner, repo, state);
    return github_get(url, token);
}

// Get specific pull request
char* nl_github_get_pr(const char* owner, const char* repo, int64_t pr_number, const char* token) {
    char url[MAX_URL_LEN];
    snprintf(url, sizeof(url), "%s/repos/%s/%s/pulls/%lld", GITHUB_API_BASE, owner, repo, (long long)pr_number);
    return github_get(url, token);
}

// Create pull request
char* nl_github_create_pr(const char* owner, const char* repo, const char* title, const char* body,
                           const char* head, const char* base, const char* token) {
    char url[MAX_URL_LEN];
    char json[MAX_JSON_LEN];

    snprintf(url, sizeof(url), "%s/repos/%s/%s/pulls", GITHUB_API_BASE, owner, repo);
    snprintf(json, sizeof(json), "{\"title\":\"%s\",\"body\":\"%s\",\"head\":\"%s\",\"base\":\"%s\"}",
             title, body, head, base);

    return github_post(url, json, token);
}

// Get GitHub token from environment variable
char* nl_github_get_token_from_env(void) {
    const char* token = getenv("GITHUB_TOKEN");
    if (token) {
        return strdup(token);
    }
    return strdup("");
}

// Check rate limit remaining
int64_t nl_github_check_rate_limit(const char* token) {
    char url[MAX_URL_LEN];
    snprintf(url, sizeof(url), "%s/rate_limit", GITHUB_API_BASE);

    char* response = github_get(url, token);

    // Simple parsing - look for "remaining":
    const char* remaining_str = strstr(response, "\"remaining\":");
    int64_t remaining = -1;
    if (remaining_str) {
        sscanf(remaining_str + 12, "%lld", (long long*)&remaining);
    }

    free(response);
    return remaining;
}
