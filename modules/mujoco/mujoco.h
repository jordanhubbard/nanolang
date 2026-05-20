#ifndef NANOLANG_MUJOCO_MODULE_H
#define NANOLANG_MUJOCO_MODULE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int64_t nl_mj_available(void);
const char *nl_mj_last_error(void);
const char *nl_mj_version_string(void);
int64_t nl_mj_version(void);

void *nl_mj_load(const char *path);
void nl_mj_free(void *sim);
int64_t nl_mj_valid(void *sim);
void nl_mj_reset(void *sim);
void nl_mj_forward(void *sim);
void nl_mj_step(void *sim);
void nl_mj_step_n(void *sim, int64_t count);

double nl_mj_time(void *sim);
double nl_mj_timestep(void *sim);
void nl_mj_set_timestep(void *sim, double dt);

int64_t nl_mj_nq(void *sim);
int64_t nl_mj_nv(void *sim);
int64_t nl_mj_nu(void *sim);
int64_t nl_mj_nbody(void *sim);
int64_t nl_mj_ngeom(void *sim);
int64_t nl_mj_nsite(void *sim);
int64_t nl_mj_njoint(void *sim);

int64_t nl_mj_body_id(void *sim, const char *name);
int64_t nl_mj_geom_id(void *sim, const char *name);
int64_t nl_mj_site_id(void *sim, const char *name);
int64_t nl_mj_joint_id(void *sim, const char *name);
int64_t nl_mj_actuator_id(void *sim, const char *name);
const char *nl_mj_body_name(void *sim, int64_t id);
const char *nl_mj_geom_name(void *sim, int64_t id);

double nl_mj_body_x(void *sim, int64_t id);
double nl_mj_body_y(void *sim, int64_t id);
double nl_mj_body_z(void *sim, int64_t id);
double nl_mj_body_quat(void *sim, int64_t id, int64_t axis);
double nl_mj_geom_x(void *sim, int64_t id);
double nl_mj_geom_y(void *sim, int64_t id);
double nl_mj_geom_z(void *sim, int64_t id);
double nl_mj_geom_xmat(void *sim, int64_t id, int64_t axis);
double nl_mj_geom_size(void *sim, int64_t id, int64_t axis);
double nl_mj_geom_rgba(void *sim, int64_t id, int64_t axis);
int64_t nl_mj_geom_type(void *sim, int64_t id);
int64_t nl_mj_geom_body(void *sim, int64_t id);
double nl_mj_site_x(void *sim, int64_t id);
double nl_mj_site_y(void *sim, int64_t id);
double nl_mj_site_z(void *sim, int64_t id);

double nl_mj_qpos(void *sim, int64_t index);
void nl_mj_set_qpos(void *sim, int64_t index, double value);
double nl_mj_qvel(void *sim, int64_t index);
void nl_mj_set_qvel(void *sim, int64_t index, double value);
double nl_mj_ctrl(void *sim, int64_t index);
void nl_mj_set_ctrl(void *sim, int64_t index, double value);

#ifdef __cplusplus
}
#endif

#endif
