#include "pybridge.h"

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cJSON.h"

#define PYBRIDGE_PROTOCOL_VERSION 1
#define PYBRIDGE_MAX_MESSAGE (64 * 1024 * 1024)

#define PYBRIDGE_ERR_NOT_RUNNING -32000
#define PYBRIDGE_ERR_IO -32001
#define PYBRIDGE_ERR_PROTOCOL -32002
#define PYBRIDGE_ERR_SPAWN -32003
#define PYBRIDGE_ERR_ENV -32004

typedef struct {
    pid_t pid;
    int stdin_fd;
    int stdout_fd;
    int running;
    int64_t next_id;
    int log_enabled;
} PyBridgeState;

static PyBridgeState g_pybridge = {0};

static const char* pybridge_strdup_or_empty(const char* s) {
    if (!s) s = "";
    char* out = strdup(s);
    return out ? out : strdup("");
}

static void pybridge_log_line(const char* prefix, const char* payload) {
    if (!g_pybridge.log_enabled) return;
    fprintf(stderr, "[pybridge] %s: %s\n", prefix ? prefix : "log", payload ? payload : "");
}

static int pybridge_mkdir_p(const char* path) {
    if (!path || !path[0]) return -1;
    char tmp[PATH_MAX];
    size_t len = strnlen(path, sizeof(tmp));
    if (len == 0 || len >= sizeof(tmp)) return -1;
    memcpy(tmp, path, len + 1);
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';

    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

static uint64_t pybridge_fnv1a(const unsigned char* data, size_t len) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t pybridge_hash_requirements(const char* req_json) {
    if (!req_json) req_json = "";
    return pybridge_fnv1a((const unsigned char*)req_json, strlen(req_json));
}

static int pybridge_run_command(const char* command) {
    if (!command) return -1;
    int status = system(command);
    if (status == -1) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

static int pybridge_write_requirements(const char* path, const char* req_json) {
    if (!path || !req_json || !req_json[0]) return 0;
    FILE* fp = fopen(path, "w");
    if (!fp) return -1;

    cJSON* root = cJSON_Parse(req_json);
    if (root && cJSON_IsArray(root)) {
        int count = cJSON_GetArraySize(root);
        for (int i = 0; i < count; i++) {
            cJSON* item = cJSON_GetArrayItem(root, i);
            if (cJSON_IsString(item) && item->valuestring) {
                fprintf(fp, "%s\n", item->valuestring);
            }
        }
    } else if (root && cJSON_IsString(root) && root->valuestring) {
        fprintf(fp, "%s\n", root->valuestring);
    } else {
        fprintf(fp, "%s\n", req_json);
    }
    if (root) cJSON_Delete(root);
    fclose(fp);
    return 0;
}

static int pybridge_ensure_venv(const char* req_json, const char* python_bin, char* out_venv, size_t out_venv_size) {
    const char* home = getenv("HOME");
    if (!home || !home[0]) return -1;

    uint64_t hash = pybridge_hash_requirements(req_json);
    char base_dir[PATH_MAX];
    char venv_dir[PATH_MAX];
    char cache_dir[PATH_MAX];

    snprintf(base_dir, sizeof(base_dir), "%s/.nanolang/python", home);
    snprintf(venv_dir, sizeof(venv_dir), "%s/venvs/%016llx", base_dir, (unsigned long long)hash);
    snprintf(cache_dir, sizeof(cache_dir), "%s/wheel-cache", base_dir);

    if (pybridge_mkdir_p(venv_dir) != 0) return -1;
    if (pybridge_mkdir_p(cache_dir) != 0) return -1;

    char venv_python[PATH_MAX];
    snprintf(venv_python, sizeof(venv_python), "%s/bin/python", venv_dir);

    bool need_create = (access(venv_python, X_OK) != 0);
    if (need_create) {
        char cmd[PATH_MAX * 2];
        snprintf(cmd, sizeof(cmd), "\"%s\" -m venv \"%s\"", python_bin, venv_dir);
        if (pybridge_run_command(cmd) != 0) return -1;

        snprintf(cmd, sizeof(cmd), "\"%s\" -m ensurepip --default-pip", venv_python);
        if (pybridge_run_command(cmd) != 0) return -1;

        if (req_json && req_json[0]) {
            char req_file[PATH_MAX];
            snprintf(req_file, sizeof(req_file), "%s/requirements.txt", venv_dir);
            if (pybridge_write_requirements(req_file, req_json) != 0) return -1;

            snprintf(cmd, sizeof(cmd),
                     "PIP_CACHE_DIR=\"%s\" \"%s\" -m pip install --disable-pip-version-check -r \"%s\"",
                     cache_dir, venv_python, req_file);
            if (pybridge_run_command(cmd) != 0) return -1;
        }
    }

    if (out_venv && out_venv_size > 0) {
        snprintf(out_venv, out_venv_size, "%s", venv_dir);
    }
    return 0;
}

static int pybridge_write_all(int fd, const unsigned char* buf, size_t len) {
    size_t offset = 0;
    while (offset < len) {
        ssize_t written = write(fd, buf + offset, len - offset);
        if (written < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        offset += (size_t)written;
    }
    return 0;
}

static int pybridge_read_all(int fd, unsigned char* buf, size_t len) {
    size_t offset = 0;
    while (offset < len) {
        ssize_t count = read(fd, buf + offset, len - offset);
        if (count == 0) return -1;
        if (count < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        offset += (size_t)count;
    }
    return 0;
}

static int pybridge_send_frame(const char* payload, size_t len) {
    if (!payload || len > PYBRIDGE_MAX_MESSAGE) return -1;
    uint32_t be_len = htonl((uint32_t)len);
    if (pybridge_write_all(g_pybridge.stdin_fd, (const unsigned char*)&be_len, sizeof(be_len)) != 0) return -1;
    return pybridge_write_all(g_pybridge.stdin_fd, (const unsigned char*)payload, len);
}

static char* pybridge_read_frame(size_t* out_len) {
    uint32_t be_len = 0;
    if (pybridge_read_all(g_pybridge.stdout_fd, (unsigned char*)&be_len, sizeof(be_len)) != 0) return NULL;
    uint32_t len = ntohl(be_len);
    if (len == 0 || len > PYBRIDGE_MAX_MESSAGE) return NULL;

    char* buf = (char*)malloc(len + 1);
    if (!buf) return NULL;
    if (pybridge_read_all(g_pybridge.stdout_fd, (unsigned char*)buf, len) != 0) {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    return buf;
}

static cJSON* pybridge_error_response(int64_t code, const char* message, const char* traceback) {
    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "v", PYBRIDGE_PROTOCOL_VERSION);
    cJSON_AddNumberToObject(root, "id", 0);

    cJSON* err = cJSON_CreateObject();
    cJSON_AddNumberToObject(err, "code", code);
    cJSON_AddStringToObject(err, "message", message ? message : "");
    cJSON_AddStringToObject(err, "traceback", traceback ? traceback : "");
    cJSON_AddItemToObject(root, "error", err);
    return root;
}

static cJSON* pybridge_send_request(const char* op, cJSON* params) {
    if (!g_pybridge.running) {
        if (params) cJSON_Delete(params);
        return pybridge_error_response(PYBRIDGE_ERR_NOT_RUNNING, "pybridge not initialized", "");
    }

    int64_t request_id = g_pybridge.next_id++;
    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "v", PYBRIDGE_PROTOCOL_VERSION);
    cJSON_AddNumberToObject(root, "id", request_id);
    cJSON_AddStringToObject(root, "op", op ? op : "");
    if (params) {
        cJSON_AddItemToObject(root, "params", params);
    } else {
        cJSON_AddItemToObject(root, "params", cJSON_CreateObject());
    }

    char* payload = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!payload) return pybridge_error_response(PYBRIDGE_ERR_PROTOCOL, "failed to encode request", "");

    pybridge_log_line("request", payload);

    if (pybridge_send_frame(payload, strlen(payload)) != 0) {
        free(payload);
        return pybridge_error_response(PYBRIDGE_ERR_IO, "failed to write request", "");
    }
    free(payload);

    size_t response_len = 0;
    char* response_payload = pybridge_read_frame(&response_len);
    if (!response_payload) return pybridge_error_response(PYBRIDGE_ERR_IO, "failed to read response", "");

    pybridge_log_line("response", response_payload);

    cJSON* response = cJSON_Parse(response_payload);
    free(response_payload);
    if (!response) return pybridge_error_response(PYBRIDGE_ERR_PROTOCOL, "invalid response JSON", "");
    return response;
}

static int pybridge_spawn_helper(const char* python_bin, const char* helper_path, int64_t privileged) {
    int stdin_pipe[2];
    int stdout_pipe[2];
    if (pipe(stdin_pipe) != 0) return -1;
    if (pipe(stdout_pipe) != 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return -1;
    }

    if (pid == 0) {
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        setenv("PYTHONUNBUFFERED", "1", 1);
        if (g_pybridge.log_enabled) setenv("PYBRIDGE_LOG", "1", 1);
        if (privileged) setenv("PYBRIDGE_PRIVILEGED", "1", 1);

        execl(python_bin, python_bin, "-u", helper_path, "--protocol", "1", (char*)NULL);
        _exit(127);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    g_pybridge.pid = pid;
    g_pybridge.stdin_fd = stdin_pipe[1];
    g_pybridge.stdout_fd = stdout_pipe[0];
    g_pybridge.running = 1;
    g_pybridge.next_id = 1;

    return 0;
}

int64_t nl_pybridge_init(const char* requirements_json, int64_t privileged) {
    if (g_pybridge.running) return 1;

    const char* log_env = getenv("NANOLANG_PYBRIDGE_LOG");
    if (!log_env || !log_env[0]) log_env = getenv("PYBRIDGE_LOG");
    g_pybridge.log_enabled = (log_env && log_env[0] && strcmp(log_env, "0") != 0) ? 1 : 0;

    const char* python_bin = getenv("NANOLANG_PYBRIDGE_PYTHON");
    if (!python_bin || !python_bin[0]) python_bin = "python3";

    const char* helper_path = getenv("NANOLANG_PYBRIDGE_HELPER");
    if (!helper_path || !helper_path[0]) helper_path = "tools/nanolang_pybridge.py";

    char venv_dir[PATH_MAX];
    if (pybridge_ensure_venv(requirements_json, python_bin, venv_dir, sizeof(venv_dir)) != 0) {
        return 0;
    }

    char venv_python[PATH_MAX];
    snprintf(venv_python, sizeof(venv_python), "%s/bin/python", venv_dir);
    const char* spawn_python = venv_python;

    if (pybridge_spawn_helper(spawn_python, helper_path, privileged) != 0) {
        g_pybridge.running = 0;
        return 0;
    }

    cJSON* params = cJSON_CreateObject();
    cJSON_AddStringToObject(params, "client", "nanolang");
    cJSON_AddNumberToObject(params, "protocol", PYBRIDGE_PROTOCOL_VERSION);
    cJSON_AddBoolToObject(params, "privileged", privileged ? 1 : 0);

    cJSON* response = pybridge_send_request("hello", params);
    if (!response) {
        nl_pybridge_shutdown();
        return 0;
    }

    cJSON* err = cJSON_GetObjectItem(response, "error");
    if (err) {
        cJSON_Delete(response);
        nl_pybridge_shutdown();
        return 0;
    }

    cJSON_Delete(response);
    return 1;
}

const char* nl_pybridge_request(const char* op, const char* params_json) {
    cJSON* params = NULL;
    if (params_json && params_json[0]) {
        params = cJSON_Parse(params_json);
    }
    if (!params) params = cJSON_CreateObject();

    cJSON* response = pybridge_send_request(op, params);
    if (!response) return pybridge_strdup_or_empty("");

    char* payload = cJSON_PrintUnformatted(response);
    cJSON_Delete(response);
    if (!payload) return pybridge_strdup_or_empty("");
    return payload;
}

static int64_t pybridge_extract_handle(cJSON* response) {
    if (!response) return -1;
    cJSON* err = cJSON_GetObjectItem(response, "error");
    if (err) return -1;
    cJSON* result = cJSON_GetObjectItem(response, "result");
    if (!result) return -1;
    cJSON* handle = cJSON_GetObjectItem(result, "handle");
    if (!handle || !cJSON_IsNumber(handle)) return -1;
    return (int64_t)handle->valuedouble;
}

int64_t nl_pybridge_import(const char* module_name) {
    cJSON* params = cJSON_CreateObject();
    cJSON_AddStringToObject(params, "module", module_name ? module_name : "");

    cJSON* response = pybridge_send_request("import", params);
    int64_t handle = pybridge_extract_handle(response);
    if (response) cJSON_Delete(response);
    return handle;
}

const char* nl_pybridge_call(int64_t handle, const char* method, const char* args_json, const char* kwargs_json) {
    cJSON* params = cJSON_CreateObject();
    cJSON_AddNumberToObject(params, "handle", (double)handle);
    cJSON_AddStringToObject(params, "method", method ? method : "");

    cJSON* args = NULL;
    if (args_json && args_json[0]) args = cJSON_Parse(args_json);
    if (!args) args = cJSON_CreateArray();
    cJSON_AddItemToObject(params, "args", args);

    cJSON* kwargs = NULL;
    if (kwargs_json && kwargs_json[0]) kwargs = cJSON_Parse(kwargs_json);
    if (!kwargs) kwargs = cJSON_CreateObject();
    cJSON_AddItemToObject(params, "kwargs", kwargs);

    cJSON* response = pybridge_send_request("call", params);
    if (!response) return pybridge_strdup_or_empty("");

    char* payload = cJSON_PrintUnformatted(response);
    cJSON_Delete(response);
    if (!payload) return pybridge_strdup_or_empty("");
    return payload;
}

const char* nl_pybridge_get(int64_t handle, const char* attr) {
    cJSON* params = cJSON_CreateObject();
    cJSON_AddNumberToObject(params, "handle", (double)handle);
    cJSON_AddStringToObject(params, "attr", attr ? attr : "");

    cJSON* response = pybridge_send_request("getattr", params);
    if (!response) return pybridge_strdup_or_empty("");

    char* payload = cJSON_PrintUnformatted(response);
    cJSON_Delete(response);
    if (!payload) return pybridge_strdup_or_empty("");
    return payload;
}

int64_t nl_pybridge_set(int64_t handle, const char* attr, const char* value_json) {
    cJSON* params = cJSON_CreateObject();
    cJSON_AddNumberToObject(params, "handle", (double)handle);
    cJSON_AddStringToObject(params, "attr", attr ? attr : "");

    cJSON* value = NULL;
    if (value_json && value_json[0]) value = cJSON_Parse(value_json);
    if (!value) value = cJSON_CreateNull();
    cJSON_AddItemToObject(params, "value", value);

    cJSON* response = pybridge_send_request("setattr", params);
    if (!response) return 0;

    cJSON* err = cJSON_GetObjectItem(response, "error");
    int64_t ok = err ? 0 : 1;
    cJSON_Delete(response);
    return ok;
}

int64_t nl_pybridge_release(int64_t handle) {
    cJSON* params = cJSON_CreateObject();
    cJSON_AddNumberToObject(params, "handle", (double)handle);

    cJSON* response = pybridge_send_request("release", params);
    if (!response) return 0;

    cJSON* err = cJSON_GetObjectItem(response, "error");
    int64_t ok = err ? 0 : 1;
    cJSON_Delete(response);
    return ok;
}

const char* nl_pybridge_ping(void) {
    cJSON* params = cJSON_CreateObject();
    cJSON* response = pybridge_send_request("ping", params);
    if (!response) return pybridge_strdup_or_empty("");

    char* payload = cJSON_PrintUnformatted(response);
    cJSON_Delete(response);
    if (!payload) return pybridge_strdup_or_empty("");
    return payload;
}

const char* nl_pybridge_sysinfo(void) {
    cJSON* params = cJSON_CreateObject();
    cJSON* response = pybridge_send_request("sysinfo", params);
    if (!response) return pybridge_strdup_or_empty("");

    char* payload = cJSON_PrintUnformatted(response);
    cJSON_Delete(response);
    if (!payload) return pybridge_strdup_or_empty("");
    return payload;
}

const char* nl_pybridge_deps(void) {
    cJSON* params = cJSON_CreateObject();
    cJSON* response = pybridge_send_request("deps", params);
    if (!response) return pybridge_strdup_or_empty("");

    char* payload = cJSON_PrintUnformatted(response);
    cJSON_Delete(response);
    if (!payload) return pybridge_strdup_or_empty("");
    return payload;
}

void nl_pybridge_shutdown(void) {
    if (!g_pybridge.running) return;

    cJSON* params = cJSON_CreateObject();
    cJSON* response = pybridge_send_request("shutdown", params);
    if (response) cJSON_Delete(response);

    close(g_pybridge.stdin_fd);
    close(g_pybridge.stdout_fd);

    int status = 0;
    waitpid(g_pybridge.pid, &status, 0);

    g_pybridge.running = 0;
    g_pybridge.pid = 0;
    g_pybridge.stdin_fd = -1;
    g_pybridge.stdout_fd = -1;
    g_pybridge.log_enabled = 0;
}
