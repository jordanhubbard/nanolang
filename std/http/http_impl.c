/*
 * HTTP Client Implementation for Nanolang
 * Wraps libcurl for HTTP requests
 */

#include <curl/curl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

typedef struct {
    char* body;
    size_t body_size;
    long status_code;
    char* error_message;
    struct curl_slist* response_headers;
    int success;
} nl_http_response_t;

// Callback for receiving response body
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t real_size = size * nmemb;
    nl_http_response_t* response = (nl_http_response_t*)userp;
    
    char* new_body = realloc(response->body, response->body_size + real_size + 1);
    if (!new_body) return 0;
    
    response->body = new_body;
    memcpy(&(response->body[response->body_size]), contents, real_size);
    response->body_size += real_size;
    response->body[response->body_size] = '\0';
    
    return real_size;
}

// Callback for receiving response headers
static size_t header_callback(char* buffer, size_t size, size_t nitems, void* userdata) {
    size_t real_size = size * nitems;
    nl_http_response_t* response = (nl_http_response_t*)userdata;
    
    // Store header line
    char* header_line = strndup(buffer, real_size);
    if (header_line) {
        response->response_headers = curl_slist_append(response->response_headers, header_line);
        free(header_line);
    }
    
    return real_size;
}

// Create new response object
static nl_http_response_t* create_response(void) {
    nl_http_response_t* response = calloc(1, sizeof(nl_http_response_t));
    if (!response) return NULL;
    
    response->body = malloc(1);
    if (!response->body) {
        free(response);
        return NULL;
    }
    response->body[0] = '\0';
    response->body_size = 0;
    response->status_code = 0;
    response->error_message = NULL;
    response->response_headers = NULL;
    response->success = 0;
    
    return response;
}

// Perform HTTP request
static void* perform_request(const char* url, const char* method, 
                             const char* data, const char* headers_json) {
    nl_http_response_t* response = create_response();
    if (!response) return NULL;
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        response->error_message = strdup("Failed to initialize CURL");
        return response;
    }
    
    // Set URL
    curl_easy_setopt(curl, CURLOPT_URL, url);
    
    // Set method
    if (strcmp(method, "POST") == 0) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (data) curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
    } else if (strcmp(method, "PUT") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        if (data) curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
    } else if (strcmp(method, "DELETE") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    } else if (strcmp(method, "PATCH") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PATCH");
        if (data) curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
    } else if (strcmp(method, "HEAD") == 0) {
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    }
    // GET is default
    
    // Set headers (simplified - expects JSON but doesn't parse it yet)
    struct curl_slist* headers = NULL;
    if (headers_json && strlen(headers_json) > 0) {
        // Simple header parsing (would need JSON lib for proper implementation)
        // For now, accept "Content-Type: application/json" style strings
        headers = curl_slist_append(headers, headers_json);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    // Set callbacks
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, response);
    
    // Follow redirects
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response->status_code);
        response->success = 1;
    } else {
        response->error_message = strdup(curl_easy_strerror(res));
        response->success = 0;
    }
    
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return response;
}

// --- Public API ---

void* nl_http_get(const char* url) {
    return perform_request(url, "GET", NULL, NULL);
}

void* nl_http_post(const char* url, const char* data) {
    return perform_request(url, "POST", data, NULL);
}

void* nl_http_put(const char* url, const char* data) {
    return perform_request(url, "PUT", data, NULL);
}

void* nl_http_delete(const char* url) {
    return perform_request(url, "DELETE", NULL, NULL);
}

void* nl_http_patch(const char* url, const char* data) {
    return perform_request(url, "PATCH", data, NULL);
}

void* nl_http_head(const char* url) {
    return perform_request(url, "HEAD", NULL, NULL);
}

void* nl_http_get_with_headers(const char* url, const char* headers) {
    return perform_request(url, "GET", NULL, headers);
}

void* nl_http_post_with_headers(const char* url, const char* data, const char* headers) {
    return perform_request(url, "POST", data, headers);
}

// --- Response Accessors ---

int64_t nl_http_response_status(void* response) {
    if (!response) return 0;
    nl_http_response_t* resp = (nl_http_response_t*)response;
    return resp->status_code;
}

const char* nl_http_response_body(void* response) {
    if (!response) return "";
    nl_http_response_t* resp = (nl_http_response_t*)response;
    return resp->body ? resp->body : "";
}

const char* nl_http_response_header(void* response, const char* header_name) {
    if (!response || !header_name) return "";
    
    nl_http_response_t* resp = (nl_http_response_t*)response;
    if (!resp->response_headers) return "";
    
    // Search through headers
    struct curl_slist* current = resp->response_headers;
    size_t header_len = strlen(header_name);
    
    while (current) {
        if (strncasecmp(current->data, header_name, header_len) == 0 &&
            current->data[header_len] == ':') {
            // Found header, return value (skip colon and spaces)
            const char* value = current->data + header_len + 1;
            while (*value == ' ') value++;
            return strdup(value);
        }
        current = current->next;
    }
    
    return "";
}

int64_t nl_http_response_ok(void* response) {
    if (!response) return 0;
    nl_http_response_t* resp = (nl_http_response_t*)response;
    return (resp->success && resp->status_code >= 200 && resp->status_code < 300) ? 1 : 0;
}

const char* nl_http_response_error(void* response) {
    if (!response) return "";
    nl_http_response_t* resp = (nl_http_response_t*)response;
    return resp->error_message ? resp->error_message : "";
}

void nl_http_free_response(void* response) {
    if (!response) return;
    
    nl_http_response_t* resp = (nl_http_response_t*)response;
    if (resp->body) free(resp->body);
    if (resp->error_message) free(resp->error_message);
    if (resp->response_headers) curl_slist_free_all(resp->response_headers);
    free(resp);
}

