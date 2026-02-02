# Chapter 8: LLM-Powered Profiling

## Overview

NanoLang features an **LLM-powered profiling system** that enables self-optimizing code. When you compile with the `-pg` flag, your program automatically profiles itself and outputs **structured JSON** that LLMs can analyze to suggest performance improvements.

This is a **major innovation**: instead of manually interpreting profiler output, you can feed the JSON directly to an LLM along with your source code, and receive concrete optimization suggestions related to your actual code.

**Key Features:**
- 🤖 **LLM-Ready Output** - Structured JSON designed for AI analysis
- 🔄 **Self-Improving Loop** - Profile → Analyze → Optimize → Repeat
- 🌍 **Cross-Platform** - Works on Linux (gprofng) and macOS (sample)
- ⚡ **Zero Configuration** - Just add `-pg` flag, profiling runs automatically
- 📊 **Actionable Insights** - Maps hotspots to source code locations

## Quick Start

### 1. Compile with Profiling

```bash
# Add the -pg flag when compiling
./bin/nanoc myprogram.nano -o bin/myprogram -pg
```

### 2. Run Your Program

```bash
# Just run it normally - profiling happens automatically!
./bin/myprogram
```

### 3. Get LLM-Friendly JSON

The program outputs structured JSON to stderr on exit:

```json
{
  "profile_type": "sampling",
  "platform": "Linux",
  "tool": "gprofng",
  "binary": "./bin/myprogram",
  "hotspots": [
    {
      "function": "process_pixels",
      "samples": 3421,
      "pct_time": 68.4,
      "per_call_us": 1234.5,
      "location": "src/renderer.nano:145"
    },
    {
      "function": "calculate_lighting",
      "samples": 892,
      "pct_time": 17.8,
      "per_call_us": 234.1,
      "location": "src/lighting.nano:67"
    }
  ],
  "analysis_hints": [
    "Functions consuming >10% of runtime are optimization targets",
    "Look for O(n²) algorithms in top hotspots",
    "Consider caching or memoization for frequently-called functions"
  ]
}
```

### 4. Feed to LLM for Analysis

```bash
# Capture profiling output and send to Claude/GPT
./bin/myprogram 2> profile.json

# Then in your LLM chat:
# "Here's profiling data from my NanoLang program. What optimizations do you recommend?"
# [paste profile.json and relevant source code]
```

## How It Works

### Compilation Flags

The `-pg` flag adds these C compiler options:

| Flag | Purpose |
|------|---------|
| `-pg` | Enable profiling instrumentation |
| `-g` | Include debug symbols for readable function names |
| `-fno-omit-frame-pointer` | Accurate stack traces |
| `-fno-optimize-sibling-calls` | Clearer call chains |

### Platform-Specific Tools

NanoLang automatically uses the right tool for your OS:

**Linux:**
- Uses `gprofng collect` (from binutils)
- No special permissions required
- Sampling-based profiling (low overhead)

**macOS:**
- Uses `sample` command
- No special permissions required on modern macOS
- Sampling-based profiling (low overhead)

### Automatic Profiling Flow

```
┌─────────────────┐
│ Compile with -pg│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run program     │
│ (normal usage)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ At exit:        │
│ - Run profiler  │
│ - Parse output  │
│ - Generate JSON │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ JSON → stderr   │
│ (LLM-ready)     │
└─────────────────┘
```

## Complete Example

Let's profile a raytracer and optimize it using LLM feedback.

### Initial Implementation

```nano
<!-- SNIPPET: profiling_example_initial -->
/*
 * Simple raytracer - initial implementation
 */

struct Vec3 { x: float, y: float, z: float }
struct Ray { origin: Vec3, direction: Vec3 }
struct Sphere { center: Vec3, radius: float }

fn vec3_dot(a: Vec3, b: Vec3) -> float {
    return (+ (+ (* a.x b.x) (* a.y b.y)) (* a.z b.z))
}

fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3 {
        x: (- a.x b.x),
        y: (- a.y b.y),
        z: (- a.z b.z)
    }
}

fn ray_sphere_intersect(ray: Ray, sphere: Sphere) -> bool {
    let oc: Vec3 = (vec3_sub ray.origin sphere.center)
    let a: float = (vec3_dot ray.direction ray.direction)
    let b: float = (* 2.0 (vec3_dot oc ray.direction))
    let c: float = (- (vec3_dot oc oc) (* sphere.radius sphere.radius))
    let discriminant: float = (- (* b b) (* 4.0 (* a c)))
    return (> discriminant 0.0)
}

fn render_pixel(x: int, y: int, spheres: array<Sphere>) -> int {
    # Create ray for this pixel
    let ray: Ray = Ray {
        origin: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        direction: Vec3 { x: (to_float x), y: (to_float y), z: 1.0 }
    }

    # Check intersection with all spheres
    let sphere_count: int = (array_length spheres)
    let mut i: int = 0
    while (< i sphere_count) {
        let sphere: Sphere = (at spheres i)
        if (ray_sphere_intersect ray sphere) {
            return 1  # Hit
        }
        set i (+ i 1)
    }
    return 0  # Miss
}

fn main() -> void {
    # Create scene with 100 spheres
    let mut spheres: array<Sphere> = (array_new 100)
    let mut i: int = 0
    while (< i 100) {
        set spheres (array_push spheres Sphere {
            center: Vec3 { x: (to_float i), y: (to_float i), z: 10.0 },
            radius: 1.0
        })
        set i (+ i 1)
    }

    # Render 800x600 image
    let mut y: int = 0
    while (< y 600) {
        let mut x: int = 0
        while (< x 800) {
            let hit: int = (render_pixel x y spheres)
            set x (+ x 1)
        }
        set y (+ y 1)
    }

    print "Rendering complete"
}
<!-- END SNIPPET -->
```

### Compile and Profile

```bash
./bin/nanoc raytracer.nano -o bin/raytracer -pg
./bin/raytracer 2> profile.json
```

### Profile Output

```json
{
  "profile_type": "sampling",
  "platform": "Linux",
  "tool": "gprofng",
  "binary": "./bin/raytracer",
  "hotspots": [
    {
      "function": "ray_sphere_intersect",
      "samples": 8923,
      "pct_time": 89.2,
      "per_call_us": 0.19,
      "location": "raytracer.nano:18"
    },
    {
      "function": "vec3_dot",
      "samples": 721,
      "pct_time": 7.2,
      "per_call_us": 0.02,
      "location": "raytracer.nano:8"
    },
    {
      "function": "render_pixel",
      "samples": 245,
      "pct_time": 2.5,
      "per_call_us": 0.51,
      "location": "raytracer.nano:26"
    }
  ],
  "analysis_hints": [
    "ray_sphere_intersect dominates execution time (89.2%)",
    "Called 48,000,000 times - optimization target",
    "vec3_dot is hot but cheap per-call (0.02 μs)"
  ]
}
```

### LLM Analysis

Feed the profile and source to your LLM:

**Prompt:**
```
I'm profiling a NanoLang raytracer. Here's the profiling output and source code.
What optimizations do you recommend?

[paste profile.json and source code]
```

**LLM Response:**
```
Based on the profiling data, ray_sphere_intersect consumes 89.2% of runtime.
I see three optimization opportunities:

1. **Eliminate redundant vec3_dot call** - You compute vec3_dot(ray.direction, ray.direction)
   but ray directions are normalized (or should be), so this is always 1.0.

2. **Reduce allocations** - vec3_sub creates a Vec3 on every call. Pass by value is
   efficient for small structs like Vec3.

3. **Early exit** - Current code checks ALL spheres. Use spatial partitioning
   (BVH or grid) to check fewer spheres per ray.

Here's the optimized ray_sphere_intersect:

[optimized code...]
```

### Optimized Implementation

```nano
<!-- SNIPPET: profiling_example_optimized -->
# Optimization 1: Assume normalized ray direction (a = 1.0)
fn ray_sphere_intersect_fast(ray: Ray, sphere: Sphere) -> bool {
    let oc: Vec3 = (vec3_sub ray.origin sphere.center)
    # Removed: let a: float = (vec3_dot ray.direction ray.direction)
    let b: float = (* 2.0 (vec3_dot oc ray.direction))
    let c: float = (- (vec3_dot oc oc) (* sphere.radius sphere.radius))
    # Use a = 1.0 directly
    let discriminant: float = (- (* b b) (* 4.0 c))
    return (> discriminant 0.0)
}

# Optimization 2: Inline small vector operations
fn render_pixel_fast(x: int, y: int, spheres: array<Sphere>) -> int {
    # Inline ray creation (avoid struct allocation)
    let ray_dir_x: float = (to_float x)
    let ray_dir_y: float = (to_float y)
    let ray_dir_z: float = 1.0

    let sphere_count: int = (array_length spheres)
    let mut i: int = 0
    while (< i sphere_count) {
        let sphere: Sphere = (at spheres i)
        # Inline intersection check with direct arithmetic
        if (ray_sphere_intersect_inline ray_dir_x ray_dir_y ray_dir_z sphere) {
            return 1
        }
        set i (+ i 1)
    }
    return 0
}
<!-- END SNIPPET -->
```

### Re-profile and Verify

```bash
./bin/nanoc raytracer_optimized.nano -o bin/raytracer_optimized -pg
time ./bin/raytracer_optimized 2> profile_optimized.json

# Compare:
# Before: 2.4 seconds
# After:  0.8 seconds (3x faster!)
```

## Advanced Usage

### Profiling with Multiple Runs

For stable results, profile multiple runs:

```bash
#!/bin/bash
# Profile 10 runs and average results

for i in {1..10}; do
    ./bin/myprogram 2>> profiles.jsonl
done

# Each run appends JSON to profiles.jsonl
# Feed all profiles to LLM for statistical analysis
```

### Profiling Specific Functions

Focus on specific code sections:

```nano
<!-- SNIPPET: selective_profiling -->
fn expensive_operation() -> int {
    # This function will show up in profile
    let mut sum: int = 0
    let mut i: int = 0
    while (< i 1000000) {
        set sum (+ sum (* i i))
        set i (+ i 1)
    }
    return sum
}

shadow expensive_operation {
    # Profile this function specifically
    let result: int = (expensive_operation)
    assert (> result 0)
}
<!-- END SNIPPET -->
```

### Integrating with CI/CD

Add profiling to your continuous integration:

```yaml
# .github/workflows/profile.yml
name: Performance Profiling

on: [push]

jobs:
  profile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Compile with profiling
        run: ./bin/nanoc src/main.nano -o bin/app -pg

      - name: Run and profile
        run: ./bin/app 2> profile.json

      - name: Upload profile
        uses: actions/upload-artifact@v2
        with:
          name: profile-data
          path: profile.json

      - name: Comment profile on PR
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const profile = JSON.parse(fs.readFileSync('profile.json'));
            const hotspots = profile.hotspots.slice(0, 5)
              .map(h => `- ${h.function}: ${h.pct_time}%`)
              .join('\n');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Performance Profile\n\n${hotspots}`
            });
```

## Platform-Specific Notes

### Linux

**Tool:** gprofng (from binutils 2.39+)

**Installation:**
```bash
# Usually pre-installed, but if needed:
sudo apt install binutils  # Debian/Ubuntu
sudo dnf install binutils  # Fedora
```

**Features:**
- No special permissions needed
- Low overhead sampling
- Detailed call graphs
- Hardware counter support

### macOS

**Tool:** sample command (built-in)

**Permissions:**
No special permissions required on modern macOS. The `sample` command works automatically for profiling your own processes.

```bash
# Just run your program - profiling is automatic!
./bin/myprogram

# Alternative: Use Instruments GUI for detailed analysis
open -a Instruments
# Then: File → New → Time Profiler
```

## Best Practices

### 1. Profile Representative Workloads

```bash
# ❌ Don't profile with toy data
./bin/image_processor small.jpg

# ✅ Profile with realistic data
./bin/image_processor large_4k_image.jpg
```

### 2. Warm Up Before Profiling

```nano
<!-- SNIPPET: warmup_profiling -->
fn main() -> void {
    # Warm up caches
    let mut i: int = 0
    while (< i 100) {
        process_data()
        set i (+ i 1)
    }

    # Now profile the real work
    let mut i: int = 0
    while (< i 10000) {
        process_data()
        set i (+ i 1)
    }
}
<!-- END SNIPPET -->
```

### 3. Profile Both Debug and Release

```bash
# Debug build - see ALL function calls
./bin/nanoc app.nano -o bin/app_debug -pg -g

# Release build - see optimizer impact
./bin/nanoc app.nano -o bin/app_release -pg -O3

# Compare hotspots - optimizer may inline functions
```

### 4. Use LLM Context Effectively

When asking LLM for optimization advice:

**Include:**
- ✅ Profiling JSON
- ✅ Source code of hot functions
- ✅ Input data characteristics
- ✅ Performance requirements
- ✅ Hardware constraints

**Example prompt:**
```
I'm optimizing a NanoLang image processing pipeline that must process
1000 images/second on a 4-core CPU. Here's the profiling data showing
gaussian_blur takes 78% of runtime. The images are 1920x1080 RGB.

[paste profile.json and gaussian_blur source]

What SIMD or algorithmic optimizations would you recommend?
```

### 5. Iterate and Measure

```
Profile → Optimize → Profile → Repeat
  ↓                              ↑
  └──────── Verify speedup ──────┘
```

Never assume an optimization worked - always re-profile!

## Troubleshooting

### No profiling output

**Symptom:** Program runs but no JSON appears

**Causes:**
1. Program crashed before exit (profiling runs at exit)
2. stderr redirected incorrectly
3. Profiler tool not available

**Solutions:**
```bash
# Check stderr explicitly
./bin/myprogram 2>&1 | tee output.log

# Check for gprofng/sample
which gprofng  # Linux
which sample   # macOS

# Ensure program exits normally
# Add explicit exit code
fn main() -> int {
    # ... your code ...
    return 0  # Ensures clean exit
}
```

### Permission denied (macOS) - Rare

**Symptom:** "sample failed: Permission denied"

**Note:** This error is rare on modern macOS. The `sample` command typically works without elevated privileges for profiling your own processes.

**If it does occur:**
```bash
# Check if your binary is code-signed with restrictive entitlements
codesign -d --entitlements - ./bin/myprogram

# Use Instruments GUI as alternative
open -a Instruments
```

### Profiling overhead too high

**Symptom:** Program much slower with -pg

**Solution:**
```bash
# Use sampling instead of instrumentation
# (NanoLang uses sampling by default, but if you added other flags)

# Reduce sample frequency (if using manual sampling)
sample <pid> 5 -file profile.txt  # Sample for 5 seconds instead of 10
```

### Unreadable function names

**Symptom:** Profiler shows "nl_func_123" instead of real names

**Solution:**
```bash
# Ensure debug symbols are included
./bin/nanoc app.nano -o bin/app -pg -g

# Check symbols are present
nm bin/app | grep nl_
```

## JSON Schema Reference

Full schema of profiling output:

```json
{
  "profile_type": "sampling",           // Always "sampling" for -pg
  "platform": "Linux" | "macOS",        // Detected OS
  "tool": "gprofng" | "sample",         // Profiler used
  "binary": "./bin/myprogram",          // Path to profiled binary
  "hotspots": [                         // Functions sorted by time
    {
      "function": "function_name",      // Function name
      "samples": 1234,                  // Sample count
      "pct_time": 12.3,                 // % of total time
      "per_call_us": 0.456,             // Microseconds per call (if available)
      "calls": 1000000,                 // Call count (if available)
      "location": "file.nano:123"       // Source location (if available)
    }
  ],
  "analysis_hints": [                   // Guidance for LLM
    "Functions with >10% time are optimization targets",
    "Look for O(n²) algorithms in hot functions"
  ]
}
```

## Summary

**NanoLang's LLM-powered profiling enables a new development workflow:**

1. **Compile with `-pg`** - Add profiling instrumentation
2. **Run normally** - Profiling happens automatically
3. **Get JSON output** - Structured, LLM-ready format
4. **Feed to LLM** - Get concrete optimization advice
5. **Apply changes** - Implement suggested improvements
6. **Re-profile** - Verify performance gains
7. **Repeat** - Continue until performance goals met

**This is self-improving code**: the language runtime generates profiling data specifically designed for AI analysis, enabling automated performance optimization loops.

**Key advantages:**
- 🎯 **Targeted** - Profile shows exactly where time is spent
- 🤖 **Automated** - LLM translates profiling data to actionable changes
- 🔄 **Iterative** - Profile → Optimize → Profile cycle
- 📊 **Data-Driven** - No guessing, measure everything

## See Also

- **[Performance Guide](https://github.com/jordanhubbard/nanolang/blob/main/docs/PERFORMANCE.md)** - General performance optimization
- **[Debugging Guide](https://github.com/jordanhubbard/nanolang/blob/main/docs/DEBUGGING_GUIDE.md)** - Debugging techniques
- **[AGENTS.md](https://github.com/jordanhubbard/nanolang/blob/main/planning/AGENTS.md)** - Full profiling documentation
- **[examples/advanced/performance_optimization.nano](../examples/advanced/performance_optimization.nano)** - More examples

---

**Last Updated:** January 31, 2026
**NanoLang Version:** 2.0+
