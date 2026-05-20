#include "mujoco.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__has_include)
#  if __has_include(<mujoco/mujoco.h>)
#    include <mujoco/mujoco.h>
#    define NL_MJ_HAS_HEADERS 1
#  else
#    define NL_MJ_HAS_HEADERS 0
#  endif
#else
#  define NL_MJ_HAS_HEADERS 0
#endif

static char g_mj_last_error[1024] = "I have not tried to load MuJoCo yet.";

static void mj_set_error(const char *msg) {
    if (!msg) msg = "";
    snprintf(g_mj_last_error, sizeof(g_mj_last_error), "%s", msg);
}

const char *nl_mj_last_error(void) {
    return g_mj_last_error;
}

#if NL_MJ_HAS_HEADERS

#include <dlfcn.h>

typedef const char *(*PFN_mj_versionString)(void);
typedef int (*PFN_mj_version)(void);
typedef mjModel *(*PFN_mj_loadXML)(const char *, const mjVFS *, char *, int);
typedef mjData *(*PFN_mj_makeData)(const mjModel *);
typedef void (*PFN_mj_deleteModel)(mjModel *);
typedef void (*PFN_mj_deleteData)(mjData *);
typedef void (*PFN_mj_resetData)(const mjModel *, mjData *);
typedef void (*PFN_mj_forward)(const mjModel *, mjData *);
typedef void (*PFN_mj_step)(const mjModel *, mjData *);
typedef int (*PFN_mj_name2id)(const mjModel *, int, const char *);
typedef const char *(*PFN_mj_id2name)(const mjModel *, int, int);

typedef struct NLMjApi {
    void *lib;
    int loaded;
    PFN_mj_versionString version_string;
    PFN_mj_version version;
    PFN_mj_loadXML load_xml;
    PFN_mj_makeData make_data;
    PFN_mj_deleteModel delete_model;
    PFN_mj_deleteData delete_data;
    PFN_mj_resetData reset_data;
    PFN_mj_forward forward;
    PFN_mj_step step;
    PFN_mj_name2id name2id;
    PFN_mj_id2name id2name;
} NLMjApi;

typedef struct NLMjSim {
    mjModel *model;
    mjData *data;
} NLMjSim;

static NLMjApi g_api;

static void *load_symbol(const char *name) {
    void *sym = dlsym(g_api.lib, name);
    if (!sym) {
        char msg[1024];
        snprintf(msg, sizeof(msg), "I found MuJoCo but could not resolve %s.", name);
        mj_set_error(msg);
    }
    return sym;
}

static int load_api(void) {
    if (g_api.loaded) return 1;

    const char *env_path = getenv("NANOLANG_MUJOCO_LIB");
    const char *candidates[] = {
        env_path,
        "libmujoco.dylib",
        "libmujoco.so",
        "libmujoco.so.3",
        "/opt/homebrew/lib/libmujoco.dylib",
        "/usr/local/lib/libmujoco.dylib",
        "/usr/lib/libmujoco.so",
        "/usr/local/lib/libmujoco.so",
        NULL
    };

    for (int i = 0; candidates[i]; i++) {
        if (!candidates[i][0]) continue;
        g_api.lib = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (g_api.lib) break;
    }

    if (!g_api.lib) {
        const char *err = dlerror();
        char msg[1024];
        snprintf(msg, sizeof(msg),
                 "I could not load MuJoCo. Install it or set NANOLANG_MUJOCO_LIB to the shared library path. dlopen says: %s",
                 err ? err : "no loader detail");
        mj_set_error(msg);
        return 0;
    }

    g_api.version_string = (PFN_mj_versionString)load_symbol("mj_versionString");
    g_api.version = (PFN_mj_version)load_symbol("mj_version");
    g_api.load_xml = (PFN_mj_loadXML)load_symbol("mj_loadXML");
    g_api.make_data = (PFN_mj_makeData)load_symbol("mj_makeData");
    g_api.delete_model = (PFN_mj_deleteModel)load_symbol("mj_deleteModel");
    g_api.delete_data = (PFN_mj_deleteData)load_symbol("mj_deleteData");
    g_api.reset_data = (PFN_mj_resetData)load_symbol("mj_resetData");
    g_api.forward = (PFN_mj_forward)load_symbol("mj_forward");
    g_api.step = (PFN_mj_step)load_symbol("mj_step");
    g_api.name2id = (PFN_mj_name2id)load_symbol("mj_name2id");
    g_api.id2name = (PFN_mj_id2name)load_symbol("mj_id2name");

    if (!g_api.version_string || !g_api.version || !g_api.load_xml ||
        !g_api.make_data || !g_api.delete_model || !g_api.delete_data ||
        !g_api.reset_data || !g_api.forward || !g_api.step ||
        !g_api.name2id || !g_api.id2name) {
        dlclose(g_api.lib);
        memset(&g_api, 0, sizeof(g_api));
        return 0;
    }

    g_api.loaded = 1;
    mj_set_error("");
    return 1;
}

static NLMjSim *as_sim(void *sim) {
    return (NLMjSim *)sim;
}

static int valid_sim(NLMjSim *sim) {
    return sim && sim->model && sim->data;
}

static int valid_axis(int64_t axis, int64_t max) {
    return axis >= 0 && axis < max;
}

static double arr3(const mjtNum *values, int64_t id, int64_t axis) {
    if (!values || id < 0 || !valid_axis(axis, 3)) return 0.0;
    return (double)values[id * 3 + axis];
}

int64_t nl_mj_available(void) {
    return load_api() ? 1 : 0;
}

const char *nl_mj_version_string(void) {
    if (!load_api()) return "unavailable";
    return g_api.version_string();
}

int64_t nl_mj_version(void) {
    if (!load_api()) return 0;
    return (int64_t)g_api.version();
}

void *nl_mj_load(const char *path) {
    if (!path || !path[0]) {
        mj_set_error("I need a non-empty MJCF or URDF path.");
        return NULL;
    }
    if (!load_api()) return NULL;

    char error[1024] = {0};
    mjModel *model = g_api.load_xml(path, NULL, error, (int)sizeof(error));
    if (!model) {
        char msg[1024];
        snprintf(msg, sizeof(msg), "I could not load %s: %s", path, error[0] ? error : "MuJoCo returned NULL");
        mj_set_error(msg);
        return NULL;
    }

    mjData *data = g_api.make_data(model);
    if (!data) {
        g_api.delete_model(model);
        mj_set_error("I loaded the model, but MuJoCo could not allocate mjData.");
        return NULL;
    }

    NLMjSim *sim = (NLMjSim *)calloc(1, sizeof(NLMjSim));
    if (!sim) {
        g_api.delete_data(data);
        g_api.delete_model(model);
        mj_set_error("I could not allocate the NanoLang MuJoCo simulation handle.");
        return NULL;
    }

    sim->model = model;
    sim->data = data;
    g_api.forward(model, data);
    mj_set_error("");
    return sim;
}

void nl_mj_free(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    if (!sim) return;
    if (sim->data && g_api.delete_data) g_api.delete_data(sim->data);
    if (sim->model && g_api.delete_model) g_api.delete_model(sim->model);
    free(sim);
}

int64_t nl_mj_valid(void *ptr) {
    return valid_sim(as_sim(ptr)) ? 1 : 0;
}

void nl_mj_reset(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || !load_api()) return;
    g_api.reset_data(sim->model, sim->data);
    g_api.forward(sim->model, sim->data);
}

void nl_mj_forward(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || !load_api()) return;
    g_api.forward(sim->model, sim->data);
}

void nl_mj_step(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || !load_api()) return;
    g_api.step(sim->model, sim->data);
}

void nl_mj_step_n(void *ptr, int64_t count) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || !load_api() || count <= 0) return;
    for (int64_t i = 0; i < count; i++) {
        g_api.step(sim->model, sim->data);
    }
}

double nl_mj_time(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim)) return 0.0;
    return (double)sim->data->time;
}

double nl_mj_timestep(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim)) return 0.0;
    return (double)sim->model->opt.timestep;
}

void nl_mj_set_timestep(void *ptr, double dt) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || dt <= 0.0) return;
    sim->model->opt.timestep = (mjtNum)dt;
}

int64_t nl_mj_nq(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    return valid_sim(sim) ? (int64_t)sim->model->nq : 0;
}

int64_t nl_mj_nv(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    return valid_sim(sim) ? (int64_t)sim->model->nv : 0;
}

int64_t nl_mj_nu(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    return valid_sim(sim) ? (int64_t)sim->model->nu : 0;
}

int64_t nl_mj_nbody(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    return valid_sim(sim) ? (int64_t)sim->model->nbody : 0;
}

int64_t nl_mj_ngeom(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    return valid_sim(sim) ? (int64_t)sim->model->ngeom : 0;
}

int64_t nl_mj_nsite(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    return valid_sim(sim) ? (int64_t)sim->model->nsite : 0;
}

int64_t nl_mj_njoint(void *ptr) {
    NLMjSim *sim = as_sim(ptr);
    return valid_sim(sim) ? (int64_t)sim->model->njnt : 0;
}

static int64_t name_to_id(void *ptr, int type, const char *name) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || !load_api() || !name) return -1;
    return (int64_t)g_api.name2id(sim->model, type, name);
}

int64_t nl_mj_body_id(void *ptr, const char *name) {
    return name_to_id(ptr, mjOBJ_BODY, name);
}

int64_t nl_mj_geom_id(void *ptr, const char *name) {
    return name_to_id(ptr, mjOBJ_GEOM, name);
}

int64_t nl_mj_site_id(void *ptr, const char *name) {
    return name_to_id(ptr, mjOBJ_SITE, name);
}

int64_t nl_mj_joint_id(void *ptr, const char *name) {
    return name_to_id(ptr, mjOBJ_JOINT, name);
}

int64_t nl_mj_actuator_id(void *ptr, const char *name) {
    return name_to_id(ptr, mjOBJ_ACTUATOR, name);
}

static const char *id_to_name(void *ptr, int type, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || !load_api() || id < 0) return "";
    const char *name = g_api.id2name(sim->model, type, (int)id);
    return name ? name : "";
}

const char *nl_mj_body_name(void *ptr, int64_t id) {
    return id_to_name(ptr, mjOBJ_BODY, id);
}

const char *nl_mj_geom_name(void *ptr, int64_t id) {
    return id_to_name(ptr, mjOBJ_GEOM, id);
}

double nl_mj_body_x(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->nbody) return 0.0;
    return arr3(sim->data->xpos, id, 0);
}

double nl_mj_body_y(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->nbody) return 0.0;
    return arr3(sim->data->xpos, id, 1);
}

double nl_mj_body_z(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->nbody) return 0.0;
    return arr3(sim->data->xpos, id, 2);
}

double nl_mj_body_quat(void *ptr, int64_t id, int64_t axis) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->nbody || !valid_axis(axis, 4)) return 0.0;
    return (double)sim->data->xquat[id * 4 + axis];
}

double nl_mj_geom_x(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->ngeom) return 0.0;
    return arr3(sim->data->geom_xpos, id, 0);
}

double nl_mj_geom_y(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->ngeom) return 0.0;
    return arr3(sim->data->geom_xpos, id, 1);
}

double nl_mj_geom_z(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->ngeom) return 0.0;
    return arr3(sim->data->geom_xpos, id, 2);
}

double nl_mj_geom_xmat(void *ptr, int64_t id, int64_t axis) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->ngeom || !valid_axis(axis, 9)) return 0.0;
    return (double)sim->data->geom_xmat[id * 9 + axis];
}

double nl_mj_geom_size(void *ptr, int64_t id, int64_t axis) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->ngeom || !valid_axis(axis, 3)) return 0.0;
    return (double)sim->model->geom_size[id * 3 + axis];
}

double nl_mj_geom_rgba(void *ptr, int64_t id, int64_t axis) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->ngeom || !valid_axis(axis, 4)) return 0.0;
    return (double)sim->model->geom_rgba[id * 4 + axis];
}

int64_t nl_mj_geom_type(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->ngeom) return -1;
    return (int64_t)sim->model->geom_type[id];
}

int64_t nl_mj_geom_body(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->ngeom) return -1;
    return (int64_t)sim->model->geom_bodyid[id];
}

double nl_mj_site_x(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->nsite) return 0.0;
    return arr3(sim->data->site_xpos, id, 0);
}

double nl_mj_site_y(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->nsite) return 0.0;
    return arr3(sim->data->site_xpos, id, 1);
}

double nl_mj_site_z(void *ptr, int64_t id) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || id < 0 || id >= sim->model->nsite) return 0.0;
    return arr3(sim->data->site_xpos, id, 2);
}

double nl_mj_qpos(void *ptr, int64_t index) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || index < 0 || index >= sim->model->nq) return 0.0;
    return (double)sim->data->qpos[index];
}

void nl_mj_set_qpos(void *ptr, int64_t index, double value) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || index < 0 || index >= sim->model->nq) return;
    sim->data->qpos[index] = (mjtNum)value;
}

double nl_mj_qvel(void *ptr, int64_t index) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || index < 0 || index >= sim->model->nv) return 0.0;
    return (double)sim->data->qvel[index];
}

void nl_mj_set_qvel(void *ptr, int64_t index, double value) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || index < 0 || index >= sim->model->nv) return;
    sim->data->qvel[index] = (mjtNum)value;
}

double nl_mj_ctrl(void *ptr, int64_t index) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || index < 0 || index >= sim->model->nu) return 0.0;
    return (double)sim->data->ctrl[index];
}

void nl_mj_set_ctrl(void *ptr, int64_t index, double value) {
    NLMjSim *sim = as_sim(ptr);
    if (!valid_sim(sim) || index < 0 || index >= sim->model->nu) return;
    sim->data->ctrl[index] = (mjtNum)value;
}

#else

int64_t nl_mj_available(void) {
    mj_set_error("I was built without MuJoCo headers. Put mujoco/mujoco.h on the C include path and rebuild this module.");
    return 0;
}

const char *nl_mj_version_string(void) {
    return "unavailable";
}

int64_t nl_mj_version(void) {
    return 0;
}

void *nl_mj_load(const char *path) {
    (void)path;
    mj_set_error("I was built without MuJoCo headers. Put mujoco/mujoco.h on the C include path and rebuild this module.");
    return NULL;
}

void nl_mj_free(void *sim) { (void)sim; }
int64_t nl_mj_valid(void *sim) { (void)sim; return 0; }
void nl_mj_reset(void *sim) { (void)sim; }
void nl_mj_forward(void *sim) { (void)sim; }
void nl_mj_step(void *sim) { (void)sim; }
void nl_mj_step_n(void *sim, int64_t count) { (void)sim; (void)count; }

double nl_mj_time(void *sim) { (void)sim; return 0.0; }
double nl_mj_timestep(void *sim) { (void)sim; return 0.0; }
void nl_mj_set_timestep(void *sim, double dt) { (void)sim; (void)dt; }

int64_t nl_mj_nq(void *sim) { (void)sim; return 0; }
int64_t nl_mj_nv(void *sim) { (void)sim; return 0; }
int64_t nl_mj_nu(void *sim) { (void)sim; return 0; }
int64_t nl_mj_nbody(void *sim) { (void)sim; return 0; }
int64_t nl_mj_ngeom(void *sim) { (void)sim; return 0; }
int64_t nl_mj_nsite(void *sim) { (void)sim; return 0; }
int64_t nl_mj_njoint(void *sim) { (void)sim; return 0; }

int64_t nl_mj_body_id(void *sim, const char *name) { (void)sim; (void)name; return -1; }
int64_t nl_mj_geom_id(void *sim, const char *name) { (void)sim; (void)name; return -1; }
int64_t nl_mj_site_id(void *sim, const char *name) { (void)sim; (void)name; return -1; }
int64_t nl_mj_joint_id(void *sim, const char *name) { (void)sim; (void)name; return -1; }
int64_t nl_mj_actuator_id(void *sim, const char *name) { (void)sim; (void)name; return -1; }
const char *nl_mj_body_name(void *sim, int64_t id) { (void)sim; (void)id; return ""; }
const char *nl_mj_geom_name(void *sim, int64_t id) { (void)sim; (void)id; return ""; }

double nl_mj_body_x(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }
double nl_mj_body_y(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }
double nl_mj_body_z(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }
double nl_mj_body_quat(void *sim, int64_t id, int64_t axis) { (void)sim; (void)id; (void)axis; return 0.0; }
double nl_mj_geom_x(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }
double nl_mj_geom_y(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }
double nl_mj_geom_z(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }
double nl_mj_geom_xmat(void *sim, int64_t id, int64_t axis) { (void)sim; (void)id; (void)axis; return 0.0; }
double nl_mj_geom_size(void *sim, int64_t id, int64_t axis) { (void)sim; (void)id; (void)axis; return 0.0; }
double nl_mj_geom_rgba(void *sim, int64_t id, int64_t axis) { (void)sim; (void)id; (void)axis; return 0.0; }
int64_t nl_mj_geom_type(void *sim, int64_t id) { (void)sim; (void)id; return -1; }
int64_t nl_mj_geom_body(void *sim, int64_t id) { (void)sim; (void)id; return -1; }
double nl_mj_site_x(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }
double nl_mj_site_y(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }
double nl_mj_site_z(void *sim, int64_t id) { (void)sim; (void)id; return 0.0; }

double nl_mj_qpos(void *sim, int64_t index) { (void)sim; (void)index; return 0.0; }
void nl_mj_set_qpos(void *sim, int64_t index, double value) { (void)sim; (void)index; (void)value; }
double nl_mj_qvel(void *sim, int64_t index) { (void)sim; (void)index; return 0.0; }
void nl_mj_set_qvel(void *sim, int64_t index, double value) { (void)sim; (void)index; (void)value; }
double nl_mj_ctrl(void *sim, int64_t index) { (void)sim; (void)index; return 0.0; }
void nl_mj_set_ctrl(void *sim, int64_t index, double value) { (void)sim; (void)index; (void)value; }

#endif
