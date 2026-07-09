# Bullet Physics Module for NanoLang

Real-time soft body and rigid body physics simulation using the [Bullet Physics SDK](https://github.com/bulletphysics/bullet3).

## Features

- **Soft Body Physics**: Deformable objects with realistic material properties
- **Rigid Body Physics**: Static and dynamic solid objects
- **Collision Detection**: Automatic soft-soft and soft-rigid collision handling
- **High Performance**: Optimized C++ physics engine
- **Easy Integration**: Standard NanoLang module interface

## Installation

### macOS

```bash
brew install bullet
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install libbullet-dev
```

### Linux (Fedora/RHEL)

```bash
sudo yum install bullet-devel
```

### Build Module

```bash
cd modules/bullet
../../modules/tools/build_module.sh .
```

The build system will automatically:
- Check if Bullet is installed
- Prompt to install if missing
- Discover compilation flags via pkg-config
- Link all required Bullet libraries

## Usage

```nano
import "modules/bullet/bullet.nano" as Bullet
import "modules/sdl/sdl.nano" as SDL

fn main() -> int {
    /* Initialize physics world */
    unsafe {
        (Bullet.nl_bullet_init)
    }
    
    /* Create a soft body sphere at (0, 10, 0) with radius 1.0 */
    let sphere: int = 0
    unsafe {
        set sphere (Bullet.nl_bullet_create_soft_sphere 0.0 10.0 0.0 1.0 32)
    }
    
    /* Create a static ground plane */
    unsafe {
        (Bullet.nl_bullet_create_rigid_box 0.0 (- 0.0 5.0) 0.0 50.0 0.5 50.0 0.0 0.5)
    }
    
    /* Simulation loop */
    let mut running: bool = true
    while running {
        /* Step physics (60 FPS) */
        unsafe {
            (Bullet.nl_bullet_step 0.016666)
        }
        
        /* Get soft body node positions for rendering */
        let node_count: int = 0
        unsafe {
            set node_count (Bullet.nl_bullet_get_soft_body_node_count sphere)
        }
        
        let mut i: int = 0
        while (< i node_count) {
            let x: float = 0.0
            let y: float = 0.0
            let z: float = 0.0
            unsafe {
                set x (Bullet.nl_bullet_get_soft_body_node_x sphere i)
                set y (Bullet.nl_bullet_get_soft_body_node_y sphere i)
                set z (Bullet.nl_bullet_get_soft_body_node_z sphere i)
            }
            /* Render node... */
            set i (+ i 1)
        }
    }
    
    unsafe {
        (Bullet.nl_bullet_cleanup)
    }
    return 0
}
```

## API Reference

### World Management

- `nl_bullet_init() -> int` - Initialize physics world
- `nl_bullet_cleanup() -> void` - Cleanup all physics objects
- `nl_bullet_step(time_step: float) -> void` - Step simulation forward

### Soft Body

- `nl_bullet_create_soft_sphere(x, y, z, radius, resolution) -> int` - Create deformable sphere
- `nl_bullet_get_soft_body_node_count(handle) -> int` - Get number of simulation nodes
- `nl_bullet_get_soft_body_node_{x|y|z}(handle, node_idx) -> float` - Get node position
- `nl_bullet_remove_soft_body(handle) -> void` - Remove soft body from simulation

### Rigid Body

- `nl_bullet_create_rigid_box(x, y, z, hw, hh, hd, mass, restitution) -> int` - Create box
- `nl_bullet_create_rigid_box_rotated(..., angle_degrees, ...) -> int` - Create rotated box
- `nl_bullet_get_rigid_body_{x|y|z}(handle) -> float` - Get position
- `nl_bullet_get_rigid_body_rot_{x|y|z|w}(handle) -> float` - Get rotation quaternion

## Examples

See `examples/bullet_soft_body_beads.nano` for a complete demo showing:
- Continuous spawning of soft body spheres
- Plinko-style obstacle course
- Real-time rendering with SDL2
- Performance optimizations

## Physics Configuration

The module uses optimized settings for soft bodies:
- **Volume Conservation**: 20.0 (maintains shape)
- **Pressure**: 0.5 (internal air pressure)
- **Material Stiffness**: 0.9 (springiness)
- **Collision Margin**: Automatic SDF-based detection
- **Gravity**: -98.0 m/s² (10× Earth gravity for dramatic effect)

## Performance

- Handles 100+ simultaneous soft bodies at 60 FPS
- Each soft body has 32-64 simulation nodes
- Broad-phase collision culling with DBVT
- Sequential impulse constraint solver

## Credits

Built on [Bullet Physics SDK](https://pybullet.org/) by Erwin Coumans.

