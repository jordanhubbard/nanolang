/*
 * Bullet Physics FFI for NanoLang
 * 
 * Provides soft body and rigid body physics simulation via Bullet3 SDK
 * Focus on soft body physics for realistic deformable objects
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

/* Bullet Physics headers */
#include <btBulletDynamicsCommon.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftBody.h>

/* Global physics world state */
static btSoftRigidDynamicsWorld* g_dynamicsWorld = NULL;
static btBroadphaseInterface* g_broadphase = NULL;
static btCollisionDispatcher* g_dispatcher = NULL;
static btConstraintSolver* g_solver = NULL;
static btSoftBodyRigidBodyCollisionConfiguration* g_collisionConfiguration = NULL;
static btSoftBodyWorldInfo g_softBodyWorldInfo;

/* Object tracking */
#define MAX_OBJECTS 1000
static btRigidBody* g_rigidBodies[MAX_OBJECTS];
static btSoftBody* g_softBodies[MAX_OBJECTS];
static int g_numRigidBodies = 0;
static int g_numSoftBodies = 0;

/*
 * Initialize Bullet Physics world with soft body support
 */
extern "C" int64_t nl_bullet_init(void) {
    if (g_dynamicsWorld != NULL) {
        return 1; /* Already initialized */
    }
    
    /* Soft body collision configuration */
    g_collisionConfiguration = new btSoftBodyRigidBodyCollisionConfiguration();
    g_dispatcher = new btCollisionDispatcher(g_collisionConfiguration);
    g_broadphase = new btDbvtBroadphase();
    g_solver = new btSequentialImpulseConstraintSolver();
    
    /* Create soft rigid dynamics world */
    g_dynamicsWorld = new btSoftRigidDynamicsWorld(
        g_dispatcher, g_broadphase, g_solver, g_collisionConfiguration, NULL);
    
    /* Set gravity */
    g_dynamicsWorld->setGravity(btVector3(0, -98.0, 0)); /* 10x earth gravity for drama */
    
    /* Initialize soft body world info */
    g_softBodyWorldInfo.m_dispatcher = g_dispatcher;
    g_softBodyWorldInfo.m_broadphase = g_broadphase;
    g_softBodyWorldInfo.m_gravity = btVector3(0, -98.0, 0);
    g_softBodyWorldInfo.m_sparsesdf.Initialize();
    g_softBodyWorldInfo.air_density = 1.2;
    g_softBodyWorldInfo.water_density = 0;
    g_softBodyWorldInfo.water_offset = 0;
    g_softBodyWorldInfo.water_normal = btVector3(0, 0, 0);
    
    return 1;
}

/*
 * Create a soft body sphere (bead)
 * Returns handle (index) to the soft body
 */
extern "C" int64_t nl_bullet_create_soft_sphere(
    double x, double y, double z,    /* position */
    double radius,                    /* size */
    int64_t resolution                /* subdivision count */
) {
    if (!g_dynamicsWorld || g_numSoftBodies >= MAX_OBJECTS) {
        return -1;
    }
    
    btVector3 center(x, y, z);
    
    /* Create soft body sphere using helper */
    btSoftBody* sphere = btSoftBodyHelpers::CreateEllipsoid(
        g_softBodyWorldInfo,
        center,
        btVector3(radius, radius, radius),
        resolution
    );
    
    if (!sphere) {
        return -1;
    }
    
    /* Configure soft body material */
    sphere->m_cfg.kVC = 20;    /* Volume conservation */
    sphere->m_cfg.kDP = 0.5;   /* Damping */
    sphere->m_cfg.kDG = 0.5;   /* Drag */
    sphere->m_cfg.kLF = 0.04;  /* Lift */
    sphere->m_cfg.kPR = 0.5;   /* Pressure */
    sphere->m_cfg.kDF = 0.2;   /* Dynamic friction */
    sphere->m_cfg.kMT = 0.05;  /* Pose matching */
    sphere->m_cfg.kCHR = 1.0;  /* Rigid contacts hardness */
    sphere->m_cfg.kKHR = 0.1;  /* Kinetic contacts hardness */
    sphere->m_cfg.kSHR = 1.0;  /* Soft contacts hardness */
    sphere->m_cfg.kAHR = 0.7;  /* Anchors hardness */
    sphere->m_cfg.collisions = btSoftBody::fCollision::SDF_RS
                             | btSoftBody::fCollision::VF_SS;
    
    /* Material properties */
    sphere->m_materials[0]->m_kLST = 0.9;  /* Linear stiffness */
    sphere->m_materials[0]->m_kAST = 0.9;  /* Angular stiffness */
    sphere->m_materials[0]->m_kVST = 0.9;  /* Volume stiffness */
    
    sphere->setTotalMass(1.0, true);
    sphere->generateBendingConstraints(2);
    sphere->randomizeConstraints();
    
    /* Add to world */
    g_dynamicsWorld->addSoftBody(sphere);
    g_softBodies[g_numSoftBodies] = sphere;
    
    return g_numSoftBodies++;
}

/*
 * Create a rigid body sphere
 * Returns handle (index) to the rigid body
 */
extern "C" int64_t nl_bullet_create_rigid_sphere(
    double x, double y, double z,        /* position */
    double radius,                       /* sphere radius */
    double mass,                         /* 0 = static */
    double restitution                   /* bounciness 0-1 */
) {
    if (!g_dynamicsWorld || g_numRigidBodies >= MAX_OBJECTS) {
        return -1;
    }
    
    /* Create sphere shape */
    btCollisionShape* sphereShape = new btSphereShape(radius);
    
    /* Transform */
    btTransform transform;
    transform.setIdentity();
    transform.setOrigin(btVector3(x, y, z));
    
    /* Mass and inertia */
    btVector3 localInertia(0, 0, 0);
    if (mass > 0) {
        sphereShape->calculateLocalInertia(mass, localInertia);
    }
    
    /* Motion state */
    btDefaultMotionState* motionState = new btDefaultMotionState(transform);
    
    /* Create rigid body */
    btRigidBody::btRigidBodyConstructionInfo rbInfo(
        mass, motionState, sphereShape, localInertia);
    rbInfo.m_restitution = restitution;
    rbInfo.m_friction = 0.3;
    
    btRigidBody* body = new btRigidBody(rbInfo);
    
    /* Add to world */
    g_dynamicsWorld->addRigidBody(body);
    g_rigidBodies[g_numRigidBodies] = body;
    
    return g_numRigidBodies++;
}

/*
 * Create a rigid body box (static obstacle)
 * Returns handle (index) to the rigid body
 */
extern "C" int64_t nl_bullet_create_rigid_box(
    double x, double y, double z,        /* position */
    double half_width, double half_height, double half_depth,
    double mass,                         /* 0 = static */
    double restitution                   /* bounciness 0-1 */
) {
    if (!g_dynamicsWorld || g_numRigidBodies >= MAX_OBJECTS) {
        return -1;
    }
    
    /* Create box shape */
    btCollisionShape* boxShape = new btBoxShape(
        btVector3(half_width, half_height, half_depth));
    
    /* Transform */
    btTransform transform;
    transform.setIdentity();
    transform.setOrigin(btVector3(x, y, z));
    
    /* Mass and inertia */
    btVector3 localInertia(0, 0, 0);
    if (mass > 0) {
        boxShape->calculateLocalInertia(mass, localInertia);
    }
    
    /* Motion state */
    btDefaultMotionState* motionState = new btDefaultMotionState(transform);
    
    /* Create rigid body */
    btRigidBody::btRigidBodyConstructionInfo rbInfo(
        mass, motionState, boxShape, localInertia);
    rbInfo.m_restitution = restitution;
    rbInfo.m_friction = 0.5;
    
    btRigidBody* body = new btRigidBody(rbInfo);
    
    /* Add to world */
    g_dynamicsWorld->addRigidBody(body);
    g_rigidBodies[g_numRigidBodies] = body;
    
    return g_numRigidBodies++;
}

/*
 * Create a rotated rigid body box
 */
extern "C" int64_t nl_bullet_create_rigid_box_rotated(
    double x, double y, double z,
    double half_width, double half_height, double half_depth,
    double angle_degrees,                /* rotation angle */
    double mass,
    double restitution
) {
    if (!g_dynamicsWorld || g_numRigidBodies >= MAX_OBJECTS) {
        return -1;
    }
    
    btCollisionShape* boxShape = new btBoxShape(
        btVector3(half_width, half_height, half_depth));
    
    btTransform transform;
    transform.setIdentity();
    transform.setOrigin(btVector3(x, y, z));
    
    /* Apply Z-axis rotation */
    btQuaternion quat;
    quat.setEulerZYX(angle_degrees * M_PI / 180.0, 0, 0);
    transform.setRotation(quat);
    
    btVector3 localInertia(0, 0, 0);
    if (mass > 0) {
        boxShape->calculateLocalInertia(mass, localInertia);
    }
    
    btDefaultMotionState* motionState = new btDefaultMotionState(transform);
    btRigidBody::btRigidBodyConstructionInfo rbInfo(
        mass, motionState, boxShape, localInertia);
    rbInfo.m_restitution = restitution;
    rbInfo.m_friction = 0.5;
    
    btRigidBody* body = new btRigidBody(rbInfo);
    g_dynamicsWorld->addRigidBody(body);
    g_rigidBodies[g_numRigidBodies] = body;
    
    return g_numRigidBodies++;
}

/*
 * Step the physics simulation
 */
extern "C" void nl_bullet_step(double time_step) {
    if (g_dynamicsWorld) {
        g_dynamicsWorld->stepSimulation(time_step, 10);
    }
}

/*
 * Get soft body node count
 */
extern "C" int64_t nl_bullet_get_soft_body_node_count(int64_t handle) {
    if (handle < 0 || handle >= g_numSoftBodies || !g_softBodies[handle]) {
        return 0;
    }
    return g_softBodies[handle]->m_nodes.size();
}

/*
 * Get soft body node position (returns x, y, z via output arrays)
 * For simplicity, we'll return via multiple calls
 */
extern "C" double nl_bullet_get_soft_body_node_x(int64_t handle, int64_t node_idx) {
    if (handle < 0 || handle >= g_numSoftBodies || !g_softBodies[handle]) {
        return 0.0;
    }
    btSoftBody* body = g_softBodies[handle];
    if (node_idx < 0 || node_idx >= body->m_nodes.size()) {
        return 0.0;
    }
    return body->m_nodes[node_idx].m_x.getX();
}

extern "C" double nl_bullet_get_soft_body_node_y(int64_t handle, int64_t node_idx) {
    if (handle < 0 || handle >= g_numSoftBodies || !g_softBodies[handle]) {
        return 0.0;
    }
    btSoftBody* body = g_softBodies[handle];
    if (node_idx < 0 || node_idx >= body->m_nodes.size()) {
        return 0.0;
    }
    return body->m_nodes[node_idx].m_x.getY();
}

extern "C" double nl_bullet_get_soft_body_node_z(int64_t handle, int64_t node_idx) {
    if (handle < 0 || handle >= g_numSoftBodies || !g_softBodies[handle]) {
        return 0.0;
    }
    btSoftBody* body = g_softBodies[handle];
    if (node_idx < 0 || node_idx >= body->m_nodes.size()) {
        return 0.0;
    }
    return body->m_nodes[node_idx].m_x.getZ();
}

/*
 * Get rigid body position
 */
extern "C" double nl_bullet_get_rigid_body_x(int64_t handle) {
    if (handle < 0 || handle >= g_numRigidBodies || !g_rigidBodies[handle]) {
        return 0.0;
    }
    btTransform trans;
    g_rigidBodies[handle]->getMotionState()->getWorldTransform(trans);
    return trans.getOrigin().getX();
}

extern "C" double nl_bullet_get_rigid_body_y(int64_t handle) {
    if (handle < 0 || handle >= g_numRigidBodies || !g_rigidBodies[handle]) {
        return 0.0;
    }
    btTransform trans;
    g_rigidBodies[handle]->getMotionState()->getWorldTransform(trans);
    return trans.getOrigin().getY();
}

extern "C" double nl_bullet_get_rigid_body_z(int64_t handle) {
    if (handle < 0 || handle >= g_numRigidBodies || !g_rigidBodies[handle]) {
        return 0.0;
    }
    btTransform trans;
    g_rigidBodies[handle]->getMotionState()->getWorldTransform(trans);
    return trans.getOrigin().getZ();
}

/*
 * Get rigid body rotation (quaternion components)
 */
extern "C" double nl_bullet_get_rigid_body_rot_x(int64_t handle) {
    if (handle < 0 || handle >= g_numRigidBodies || !g_rigidBodies[handle]) {
        return 0.0;
    }
    btTransform trans;
    g_rigidBodies[handle]->getMotionState()->getWorldTransform(trans);
    return trans.getRotation().getX();
}

extern "C" double nl_bullet_get_rigid_body_rot_y(int64_t handle) {
    if (handle < 0 || handle >= g_numRigidBodies || !g_rigidBodies[handle]) {
        return 0.0;
    }
    btTransform trans;
    g_rigidBodies[handle]->getMotionState()->getWorldTransform(trans);
    return trans.getRotation().getY();
}

extern "C" double nl_bullet_get_rigid_body_rot_z(int64_t handle) {
    if (handle < 0 || handle >= g_numRigidBodies || !g_rigidBodies[handle]) {
        return 0.0;
    }
    btTransform trans;
    g_rigidBodies[handle]->getMotionState()->getWorldTransform(trans);
    return trans.getRotation().getZ();
}

extern "C" double nl_bullet_get_rigid_body_rot_w(int64_t handle) {
    if (handle < 0 || handle >= g_numRigidBodies || !g_rigidBodies[handle]) {
        return 1.0;
    }
    btTransform trans;
    g_rigidBodies[handle]->getMotionState()->getWorldTransform(trans);
    return trans.getRotation().getW();
}

/*
 * Remove soft body that fell off screen
 */
extern "C" void nl_bullet_remove_soft_body(int64_t handle) {
    if (handle < 0 || handle >= g_numSoftBodies || !g_softBodies[handle]) {
        return;
    }
    g_dynamicsWorld->removeSoftBody(g_softBodies[handle]);
    delete g_softBodies[handle];
    g_softBodies[handle] = NULL;
}

/*
 * Get total soft body count
 */
extern "C" int64_t nl_bullet_get_soft_body_count(void) {
    return g_numSoftBodies;
}

/*
 * Get total rigid body count
 */
extern "C" int64_t nl_bullet_get_rigid_body_count(void) {
    return g_numRigidBodies;
}

/*
 * Cleanup
 */
extern "C" void nl_bullet_cleanup(void) {
    if (!g_dynamicsWorld) {
        return;
    }
    
    /* Remove soft bodies */
    for (int i = 0; i < g_numSoftBodies; i++) {
        if (g_softBodies[i]) {
            g_dynamicsWorld->removeSoftBody(g_softBodies[i]);
            delete g_softBodies[i];
        }
    }
    
    /* Remove rigid bodies */
    for (int i = 0; i < g_numRigidBodies; i++) {
        if (g_rigidBodies[i]) {
            g_dynamicsWorld->removeRigidBody(g_rigidBodies[i]);
            delete g_rigidBodies[i]->getMotionState();
            delete g_rigidBodies[i]->getCollisionShape();
            delete g_rigidBodies[i];
        }
    }
    
    delete g_dynamicsWorld;
    delete g_solver;
    delete g_broadphase;
    delete g_dispatcher;
    delete g_collisionConfiguration;
    
    g_dynamicsWorld = NULL;
    g_numSoftBodies = 0;
    g_numRigidBodies = 0;
}

