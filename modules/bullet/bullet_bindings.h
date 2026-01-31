#ifndef NL_BULLET_BINDINGS_H
#define NL_BULLET_BINDINGS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int64_t nl_bullet_init(void);
void nl_bullet_cleanup(void);
void nl_bullet_step(double time_step);

/* Set gravity vector for the world (rigid + soft). Units: m/s^2 */
void nl_bullet_set_gravity(double gx, double gy, double gz);

int64_t nl_bullet_create_soft_sphere(double x, double y, double z,
                                     double radius, int64_t resolution);

int64_t nl_bullet_create_rigid_sphere(double x, double y, double z,
                                      double radius, double mass, double restitution);

int64_t nl_bullet_create_rigid_box(double x, double y, double z,
                                   double half_width, double half_height, double half_depth,
                                   double mass, double restitution);

int64_t nl_bullet_create_rigid_box_rotated(double x, double y, double z,
                                           double half_width, double half_height, double half_depth,
                                           double angle_degrees, double mass, double restitution);

int64_t nl_bullet_get_soft_body_count(void);
int64_t nl_bullet_get_soft_body_node_count(int64_t handle);
double nl_bullet_get_soft_body_node_x(int64_t handle, int64_t node_idx);
double nl_bullet_get_soft_body_node_y(int64_t handle, int64_t node_idx);
double nl_bullet_get_soft_body_node_z(int64_t handle, int64_t node_idx);
void nl_bullet_remove_soft_body(int64_t handle);

int64_t nl_bullet_get_rigid_body_count(void);
double nl_bullet_get_rigid_body_x(int64_t handle);
double nl_bullet_get_rigid_body_y(int64_t handle);
double nl_bullet_get_rigid_body_z(int64_t handle);
double nl_bullet_get_rigid_body_rot_x(int64_t handle);
double nl_bullet_get_rigid_body_rot_y(int64_t handle);
double nl_bullet_get_rigid_body_rot_z(int64_t handle);
double nl_bullet_get_rigid_body_rot_w(int64_t handle);

#ifdef __cplusplus
}
#endif

#endif /* NL_BULLET_BINDINGS_H */
