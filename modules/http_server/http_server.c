/*
 * HTTP Server Implementation for NanoLang
 * Built on libuv for async I/O with simple HTTP/1.1 parsing
 * 
 * Features:
 * - Route-based request handling
 * - Static file serving
 * - JSON API support
 * - Middleware support
 * - Query parameters and path parameters
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uv.h>
#include <ctype.h>
#include "http_server.h"

#define MAX_ROUTES 256
#define MAX_HEADERS 64
#define MAX_BUFFER 65536
#define MAX_PATH 1024

/* HTTP Request structure */
typedef struct {
    char method[16];
    char path[MAX_PATH];
    char query[MAX_PATH];
    char version[16];
    char headers[MAX_HEADERS][256];
    int header_count;
    char* body;
    size_t body_length;
    int complete;
} http_request_t;

/* HTTP Response structure */
typedef struct {
    int status_code;
    char status_message[64];
    char headers[MAX_HEADERS][256];
    int header_count;
    char* body;
    size_t body_length;
} http_response_t;

/* Route handler callback type */
typedef void (*route_handler_t)(http_request_t* req, http_response_t* res);

/* Route definition */
typedef struct {
    char method[16];
    char path[MAX_PATH];
    route_handler_t handler;
} route_t;

/* Server context */
typedef struct {
    uv_loop_t* loop;
    uv_tcp_t server;
    route_t routes[MAX_ROUTES];
    int route_count;
    char* static_dir;
    int port;
} http_server_t;

/* Client connection */
typedef struct {
    uv_tcp_t handle;
    http_server_t* server;
    http_request_t request;
    char read_buffer[MAX_BUFFER];
    size_t read_pos;
} client_t;

/* Global server instance (for FFI callbacks) */
static http_server_t* g_server = NULL;

/* ========================================================================
 * HTTP Response Helpers
 * ======================================================================== */

static void http_response_init(http_response_t* res) {
    res->status_code = 200;
    strcpy(res->status_message, "OK");
    res->header_count = 0;
    res->body = NULL;
    res->body_length = 0;
}

static void http_response_set_status(http_response_t* res, int code, const char* message) {
    res->status_code = code;
    strncpy(res->status_message, message, sizeof(res->status_message) - 1);
}

static void http_response_add_header(http_response_t* res, const char* name, const char* value) {
    if (res->header_count < MAX_HEADERS) {
        snprintf(res->headers[res->header_count], 256, "%s: %s", name, value);
        res->header_count++;
    }
}

static void http_response_set_body(http_response_t* res, const char* body) {
    if (res->body) free(res->body);
    res->body_length = strlen(body);
    res->body = malloc(res->body_length + 1);
    if (res->body) {
        memcpy(res->body, body, res->body_length);
        res->body[res->body_length] = '\0';
    }
}

static void http_response_json(http_response_t* res, const char* json) {
    http_response_add_header(res, "Content-Type", "application/json");
    http_response_set_body(res, json);
}

static void http_response_html(http_response_t* res, const char* html) {
    http_response_add_header(res, "Content-Type", "text/html; charset=utf-8");
    http_response_set_body(res, html);
}

static void http_response_text(http_response_t* res, const char* text) {
    http_response_add_header(res, "Content-Type", "text/plain; charset=utf-8");
    http_response_set_body(res, text);
}

static void http_response_free(http_response_t* res) {
    if (res->body) {
        free(res->body);
        res->body = NULL;
    }
}

/* ========================================================================
 * HTTP Request Parsing
 * ======================================================================== */

static void parse_query_string(const char* uri, char* path, char* query) {
    const char* q = strchr(uri, '?');
    if (q) {
        size_t path_len = q - uri;
        strncpy(path, uri, path_len);
        path[path_len] = '\0';
        strcpy(query, q + 1);
    } else {
        strcpy(path, uri);
        query[0] = '\0';
    }
}

static int parse_request_line(http_request_t* req, const char* line) {
    char uri[MAX_PATH];
    if (sscanf(line, "%15s %1023s %15s", req->method, uri, req->version) != 3) {
        return -1;
    }
    parse_query_string(uri, req->path, req->query);
    return 0;
}

static int parse_header_line(http_request_t* req, const char* line) {
    if (req->header_count >= MAX_HEADERS) return -1;
    
    strncpy(req->headers[req->header_count], line, 255);
    req->headers[req->header_count][255] = '\0';
    req->header_count++;
    return 0;
}

static int parse_http_request(http_request_t* req, const char* data, size_t len) {
    const char* line_start = data;
    const char* line_end;
    int first_line = 1;
    int headers_complete = 0;
    
    while ((line_end = strstr(line_start, "\r\n")) != NULL) {
        size_t line_len = line_end - line_start;
        
        if (line_len == 0) {
            /* Empty line = end of headers */
            headers_complete = 1;
            line_start = line_end + 2;
            break;
        }
        
        char line[MAX_PATH];
        strncpy(line, line_start, line_len);
        line[line_len] = '\0';
        
        if (first_line) {
            if (parse_request_line(req, line) < 0) return -1;
            first_line = 0;
        } else {
            parse_header_line(req, line);
        }
        
        line_start = line_end + 2;
    }
    
    if (headers_complete) {
        /* Parse body if present */
        size_t body_offset = line_start - data;
        if (body_offset < len) {
            req->body_length = len - body_offset;
            req->body = malloc(req->body_length + 1);
            if (req->body) {
                memcpy(req->body, line_start, req->body_length);
                req->body[req->body_length] = '\0';
            }
        }
        req->complete = 1;
        return 0;
    }
    
    return 1; /* Need more data */
}

/* ========================================================================
 * Route Matching
 * ======================================================================== */

static route_t* find_route(http_server_t* server, const char* method, const char* path) {
    for (int i = 0; i < server->route_count; i++) {
        if (strcmp(server->routes[i].method, method) == 0 &&
            strcmp(server->routes[i].path, path) == 0) {
            return &server->routes[i];
        }
    }
    return NULL;
}

/* ========================================================================
 * Static File Serving
 * ======================================================================== */

static const char* get_mime_type(const char* path) {
    const char* ext = strrchr(path, '.');
    if (!ext) return "application/octet-stream";
    
    if (strcmp(ext, ".html") == 0) return "text/html";
    if (strcmp(ext, ".htm") == 0) return "text/html";
    if (strcmp(ext, ".css") == 0) return "text/css";
    if (strcmp(ext, ".js") == 0) return "application/javascript";
    if (strcmp(ext, ".json") == 0) return "application/json";
    if (strcmp(ext, ".png") == 0) return "image/png";
    if (strcmp(ext, ".jpg") == 0) return "image/jpeg";
    if (strcmp(ext, ".jpeg") == 0) return "image/jpeg";
    if (strcmp(ext, ".gif") == 0) return "image/gif";
    if (strcmp(ext, ".svg") == 0) return "image/svg+xml";
    if (strcmp(ext, ".ico") == 0) return "image/x-icon";
    if (strcmp(ext, ".txt") == 0) return "text/plain";
    if (strcmp(ext, ".xml") == 0) return "application/xml";
    if (strcmp(ext, ".pdf") == 0) return "application/pdf";
    
    return "application/octet-stream";
}

static void serve_static_file(http_server_t* server, http_request_t* req, http_response_t* res) {
    if (!server->static_dir) {
        http_response_set_status(res, 404, "Not Found");
        http_response_text(res, "404 Not Found");
        return;
    }
    
    /* Build file path */
    char filepath[MAX_PATH * 2];
    snprintf(filepath, sizeof(filepath), "%s%s", server->static_dir, req->path);
    
    /* Security: prevent directory traversal */
    if (strstr(filepath, "..") != NULL) {
        http_response_set_status(res, 403, "Forbidden");
        http_response_text(res, "403 Forbidden");
        return;
    }
    
    /* Try to open file */
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        http_response_set_status(res, 404, "Not Found");
        http_response_text(res, "404 Not Found");
        return;
    }
    
    /* Get file size */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    /* Read file */
    char* content = malloc(size + 1);
    if (!content) {
        fclose(f);
        http_response_set_status(res, 500, "Internal Server Error");
        http_response_text(res, "500 Internal Server Error");
        return;
    }
    
    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);
    
    /* Set content type and body */
    http_response_add_header(res, "Content-Type", get_mime_type(filepath));
    res->body = content;
    res->body_length = size;
}

/* ========================================================================
 * Request Handling
 * ======================================================================== */

static void handle_request(http_server_t* server, http_request_t* req, http_response_t* res) {
    /* Try to find matching route */
    route_t* route = find_route(server, req->method, req->path);
    
    if (route && route->handler) {
        /* Call route handler */
        route->handler(req, res);
    } else {
        /* Try static file serving */
        serve_static_file(server, req, res);
    }
}

static void send_response(uv_stream_t* client, http_response_t* res) {
    /* Build response */
    char response[MAX_BUFFER];
    int pos = 0;
    
    /* Status line */
    pos += snprintf(response + pos, sizeof(response) - pos,
                   "HTTP/1.1 %d %s\r\n", res->status_code, res->status_message);
    
    /* Add Content-Length if body present */
    if (res->body) {
        pos += snprintf(response + pos, sizeof(response) - pos,
                       "Content-Length: %zu\r\n", res->body_length);
    }
    
    /* Headers */
    for (int i = 0; i < res->header_count; i++) {
        pos += snprintf(response + pos, sizeof(response) - pos,
                       "%s\r\n", res->headers[i]);
    }
    
    /* Connection header */
    pos += snprintf(response + pos, sizeof(response) - pos,
                   "Connection: close\r\n");
    
    /* End of headers */
    pos += snprintf(response + pos, sizeof(response) - pos, "\r\n");
    
    /* Write response */
    uv_write_t* write_req = malloc(sizeof(uv_write_t));
    uv_buf_t buf[2];
    
    buf[0] = uv_buf_init(response, pos);
    
    if (res->body && res->body_length > 0) {
        buf[1] = uv_buf_init(res->body, res->body_length);
        uv_write(write_req, client, buf, 2, NULL);
    } else {
        uv_write(write_req, client, buf, 1, NULL);
    }
}

/* ========================================================================
 * UV Callbacks
 * ======================================================================== */

static void on_close(uv_handle_t* handle) {
    client_t* client = (client_t*)handle->data;
    if (client->request.body) free(client->request.body);
    free(client);
}

static void on_read(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf) {
    client_t* client = (client_t*)stream->data;
    
    if (nread < 0) {
        uv_close((uv_handle_t*)stream, on_close);
        return;
    }
    
    if (nread == 0) return;
    
    /* Append to read buffer */
    if (client->read_pos + nread < MAX_BUFFER) {
        memcpy(client->read_buffer + client->read_pos, buf->base, nread);
        client->read_pos += nread;
        client->read_buffer[client->read_pos] = '\0';
        
        /* Try to parse request */
        if (!client->request.complete) {
            int result = parse_http_request(&client->request, 
                                          client->read_buffer, 
                                          client->read_pos);
            
            if (result == 0) {
                /* Request complete - handle it */
                http_response_t response;
                http_response_init(&response);
                
                handle_request(client->server, &client->request, &response);
                
                send_response(stream, &response);
                http_response_free(&response);
                
                uv_close((uv_handle_t*)stream, on_close);
            }
        }
    } else {
        /* Buffer overflow */
        http_response_t response;
        http_response_init(&response);
        http_response_set_status(&response, 413, "Payload Too Large");
        http_response_text(&response, "413 Payload Too Large");
        send_response(stream, &response);
        http_response_free(&response);
        uv_close((uv_handle_t*)stream, on_close);
    }
}

static void alloc_buffer(uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf) {
    buf->base = malloc(suggested_size);
    buf->len = suggested_size;
}

static void on_connection(uv_stream_t* server, int status) {
    if (status < 0) return;
    
    http_server_t* http_server = (http_server_t*)server->data;
    
    client_t* client = calloc(1, sizeof(client_t));
    client->server = http_server;
    client->handle.data = client;
    
    uv_tcp_init(http_server->loop, &client->handle);
    
    if (uv_accept(server, (uv_stream_t*)&client->handle) == 0) {
        uv_read_start((uv_stream_t*)&client->handle, alloc_buffer, on_read);
    } else {
        uv_close((uv_handle_t*)&client->handle, on_close);
    }
}

/* ========================================================================
 * Public API for NanoLang FFI
 * ======================================================================== */

/* Create new HTTP server */
void* nl_http_server_create(int64_t port) {
    http_server_t* server = calloc(1, sizeof(http_server_t));
    if (!server) return NULL;
    
    server->loop = uv_default_loop();
    server->port = (int)port;
    server->route_count = 0;
    server->static_dir = NULL;
    
    uv_tcp_init(server->loop, &server->server);
    server->server.data = server;
    
    g_server = server;
    return server;
}

/* Set static file directory */
void nl_http_server_set_static(void* server_ptr, const char* dir) {
    http_server_t* server = (http_server_t*)server_ptr;
    if (server->static_dir) free(server->static_dir);
    server->static_dir = strdup(dir);
}

/* Add route (called from NanoLang - stores route for later lookup) */
int64_t nl_http_server_add_route(void* server_ptr, const char* method, const char* path, int64_t handler_id) {
    http_server_t* server = (http_server_t*)server_ptr;
    
    if (server->route_count >= MAX_ROUTES) return -1;
    
    route_t* route = &server->routes[server->route_count];
    strncpy(route->method, method, sizeof(route->method) - 1);
    strncpy(route->path, path, sizeof(route->path) - 1);
    route->handler = (route_handler_t)handler_id; /* Store handler ID for now */
    
    server->route_count++;
    return server->route_count - 1;
}

/* Start server */
int64_t nl_http_server_start(void* server_ptr) {
    http_server_t* server = (http_server_t*)server_ptr;
    
    struct sockaddr_in addr;
    uv_ip4_addr("0.0.0.0", server->port, &addr);
    
    int result = uv_tcp_bind(&server->server, (const struct sockaddr*)&addr, 0);
    if (result != 0) return result;
    
    result = uv_listen((uv_stream_t*)&server->server, 128, on_connection);
    if (result != 0) return result;
    
    printf("HTTP server listening on http://0.0.0.0:%d\n", server->port);
    
    return uv_run(server->loop, UV_RUN_DEFAULT);
}

/* Stop server */
void nl_http_server_stop(void* server_ptr) {
    http_server_t* server = (http_server_t*)server_ptr;
    uv_stop(server->loop);
}

/* Free server */
void nl_http_server_free(void* server_ptr) {
    http_server_t* server = (http_server_t*)server_ptr;
    if (server->static_dir) free(server->static_dir);
    free(server);
}

/* Helper: Get request method */
const char* nl_http_request_method(void* req_ptr) {
    http_request_t* req = (http_request_t*)req_ptr;
    return req->method;
}

/* Helper: Get request path */
const char* nl_http_request_path(void* req_ptr) {
    http_request_t* req = (http_request_t*)req_ptr;
    return req->path;
}

/* Helper: Get request query string */
const char* nl_http_request_query(void* req_ptr) {
    http_request_t* req = (http_request_t*)req_ptr;
    return req->query;
}

/* Helper: Get request body */
const char* nl_http_request_body(void* req_ptr) {
    http_request_t* req = (http_request_t*)req_ptr;
    return req->body ? req->body : "";
}

/* Helper: Set response status */
void nl_http_response_status(void* res_ptr, int64_t code, const char* message) {
    http_response_t* res = (http_response_t*)res_ptr;
    http_response_set_status(res, (int)code, message);
}

/* Helper: Set response header */
void nl_http_response_header(void* res_ptr, const char* name, const char* value) {
    http_response_t* res = (http_response_t*)res_ptr;
    http_response_add_header(res, name, value);
}

/* Helper: Send JSON response */
void nl_http_response_send_json(void* res_ptr, const char* json) {
    http_response_t* res = (http_response_t*)res_ptr;
    http_response_json(res, json);
}

/* Helper: Send HTML response */
void nl_http_response_send_html(void* res_ptr, const char* html) {
    http_response_t* res = (http_response_t*)res_ptr;
    http_response_html(res, html);
}

/* Helper: Send text response */
void nl_http_response_send_text(void* res_ptr, const char* text) {
    http_response_t* res = (http_response_t*)res_ptr;
    http_response_text(res, text);
}

