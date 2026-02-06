#include "openai.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <curl/curl.h>

#define DEFAULT_API_BASE "https://api.openai.com/v1"
#define MAX_URL_LEN 1024
#define MAX_HEADER_LEN 512
#define MAX_JSON_LEN 65536

// Global configuration
static char g_api_base[MAX_URL_LEN] = DEFAULT_API_BASE;

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

// Perform HTTP POST request to OpenAI API
static char* openai_post(const char* endpoint, const char* json_body, const char* api_key) {
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

    // Build full URL
    char url[MAX_URL_LEN];
    snprintf(url, sizeof(url), "%s%s", g_api_base, endpoint);

    // Build authorization header
    char auth_header[MAX_HEADER_LEN];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, auth_header);
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

// Perform HTTP GET request to OpenAI API
static char* openai_get(const char* endpoint, const char* api_key) {
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

    // Build full URL
    char url[MAX_URL_LEN];
    snprintf(url, sizeof(url), "%s%s", g_api_base, endpoint);

    // Build authorization header
    char auth_header[MAX_HEADER_LEN];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, auth_header);

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

// ============================================================================
// Configuration functions
// ============================================================================

void nl_openai_set_api_base(const char* base_url) {
    strncpy(g_api_base, base_url, sizeof(g_api_base) - 1);
    g_api_base[sizeof(g_api_base) - 1] = '\0';
}

char* nl_openai_get_api_base(void) {
    return strdup(g_api_base);
}

char* nl_openai_get_key_from_env(void) {
    const char* key = getenv("OPENAI_API_KEY");
    if (key) {
        return strdup(key);
    }
    return strdup("");
}

// ============================================================================
// Chat completion functions
// ============================================================================

// Full chat completion with custom messages JSON
char* nl_openai_chat_completion(const char* model, const char* messages_json, const char* api_key) {
    char json[MAX_JSON_LEN];
    snprintf(json, sizeof(json), "{\"model\":\"%s\",\"messages\":%s}", model, messages_json);
    return openai_post("/chat/completions", json, api_key);
}

// Simple chat completion with system prompt and user message
char* nl_openai_chat_completion_simple(const char* model, const char* system_prompt,
                                       const char* user_message, const char* api_key) {
    char json[MAX_JSON_LEN];
    snprintf(json, sizeof(json),
             "{\"model\":\"%s\",\"messages\":[{\"role\":\"system\",\"content\":\"%s\"},{\"role\":\"user\",\"content\":\"%s\"}]}",
             model, system_prompt, user_message);
    return openai_post("/chat/completions", json, api_key);
}

// Chat completion with temperature and max_tokens
char* nl_openai_chat_completion_with_temperature(const char* model, const char* messages_json,
                                                  double temperature, int64_t max_tokens, const char* api_key) {
    char json[MAX_JSON_LEN];
    snprintf(json, sizeof(json),
             "{\"model\":\"%s\",\"messages\":%s,\"temperature\":%.2f,\"max_tokens\":%lld}",
             model, messages_json, temperature, (long long)max_tokens);
    return openai_post("/chat/completions", json, api_key);
}

// ============================================================================
// Embeddings
// ============================================================================

char* nl_openai_create_embedding(const char* model, const char* input, const char* api_key) {
    char json[MAX_JSON_LEN];
    snprintf(json, sizeof(json), "{\"model\":\"%s\",\"input\":\"%s\"}", model, input);
    return openai_post("/embeddings", json, api_key);
}

// ============================================================================
// Model management
// ============================================================================

char* nl_openai_list_models(const char* api_key) {
    return openai_get("/models", api_key);
}

char* nl_openai_get_model(const char* model_id, const char* api_key) {
    char endpoint[MAX_URL_LEN];
    snprintf(endpoint, sizeof(endpoint), "/models/%s", model_id);
    return openai_get(endpoint, api_key);
}

// ============================================================================
// Utility functions
// ============================================================================

// Rough token count estimation (1 token ≈ 4 characters)
int64_t nl_openai_count_tokens_estimate(const char* text) {
    return (int64_t)(strlen(text) / 4);
}
