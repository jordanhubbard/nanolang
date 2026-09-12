# My Package Manager

I now have a package manager. It does what a package manager should do — install dependencies, pin versions, and keep builds reproducible — without the bloat that most ecosystems have learned to tolerate.

## Overview

My packages are distributed through a Git-based registry. There is no central server to maintain, no authentication tokens to rotate, no opaque binary blobs. The registry is a Git repository. You can clone it, read it, fork it. That is the entire infrastructure.

### Key Concepts

- **`nano.toml`** — My package manifest. Declares who you are and what you need.
- **`nano.lock`** — My lockfile. Pins exact versions and checksums so builds are reproducible.
- **`nanoc-pkg`** — The CLI tool. Lives in `bin/nanoc-pkg`, implemented in `scripts/nano-pkg.sh`.
- **Registry** — A Git repository at `github.com/jordanhubbard/nano-packages`.

## Quick Start

### Create a New Package

```bash
nanoc-pkg init
```

This creates a `nano.toml` with sensible defaults. Edit it.

### Declare Dependencies

In your `nano.toml`:

```toml
[package]
name = "my_project"
version = "0.1.0"
description = "A thing I am building"

[dependencies]
vector2d = ">=1.0.0"
curl = "^1.0.0"
```

### Install Dependencies

```bash
nanoc-pkg install
```

This resolves versions from the registry, copies packages into `modules/`, and writes `nano.lock`.

### Publish a Package

```bash
nanoc-pkg publish
```

This stages your package in the local registry clone. Push it yourself when you are ready.

## Manifest Format: `nano.toml`

The manifest tells me who your package is and what it needs.

### `[package]` Section

```toml
[package]
name = "my_package"
version = "1.2.3"
description = "A concise description"
authors = ["Your Name"]
license = "MIT"
entry = "main.nano"
module_paths = ["modules", "lib"]
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Package name. Must match `[a-z][a-z0-9_]*`. |
| `version` | Yes | Semantic version: `MAJOR.MINOR.PATCH`. |
| `description` | No | One-line description. |
| `authors` | No | List of author names. |
| `license` | No | SPDX license identifier. |
| `entry` | No | Main source file (default: `main.nano`). |
| `module_paths` | No | Additional directories to search for modules. |

### `[dependencies]` Section

```toml
[dependencies]
vector2d = ">=1.0.0"      # Any version 1.0.0 or newer
curl = "^1.2.0"           # Compatible: same major, >= 1.2.0
some_lib = "~2.1.0"       # Approximate: same major.minor, >= patch
exact_lib = "3.0.0"       # Exactly this version
anything = "*"            # Any version
```

#### Version Constraint Operators

| Operator | Meaning | Example | Matches |
|----------|---------|---------|---------|
| `>=` | Greater than or equal | `>=1.2.0` | `1.2.0`, `1.3.0`, `2.0.0` |
| `^` | Compatible (same major) | `^1.2.0` | `1.2.0`, `1.9.9` — not `2.0.0` |
| `~` | Approximate (same major.minor) | `~1.2.0` | `1.2.0`, `1.2.9` — not `1.3.0` |
| (none) | Exact match | `1.2.3` | Only `1.2.3` |
| `*` | Any version | `*` | Everything |

## Lockfile: `nano.lock`

The lockfile is JSON. I chose JSON because I already parse it elsewhere and it is unambiguous.

```json
{
  "version": 1,
  "packages": {
    "vector2d": {
      "version": "1.0.0",
      "checksum": "a1b2c3d4e5f6...",
      "source": "registry"
    }
  }
}
```

### Rules

1. **Commit `nano.lock` to version control.** This is how your collaborators get the same build you have.
2. **Do not edit `nano.lock` by hand.** Let `nanoc-pkg install` manage it.
3. **Run `nanoc-pkg update` to re-resolve.** This ignores the lockfile and resolves fresh.

When `nano.lock` exists, `nanoc-pkg install` respects pinned versions as long as they still satisfy the constraints in `nano.toml`. If they do not (because you tightened a constraint), I re-resolve.

## Registry Layout

The registry is a Git repository with this structure:

```
nano-packages/
├── packages/
│   ├── vector2d/
│   │   ├── package.json        # Package metadata + version index
│   │   ├── 1.0.0/              # Version 1.0.0 contents
│   │   │   ├── nano.toml
│   │   │   ├── vector2d.nano
│   │   │   ├── module.json
│   │   │   └── README.md
│   │   └── 1.1.0/              # Version 1.1.0 contents
│   │       └── ...
│   └── curl/
│       ├── package.json
│       └── 1.0.0/
│           └── ...
└── README.md
```

Each version directory contains the full package contents — the same files you would find in `modules/<name>/` after installation.

## CLI Reference

### `nanoc-pkg install [<package>]`

Install all dependencies declared in `nano.toml`, or a specific package.

- Syncs the local registry cache (shallow git clone)
- Resolves versions against constraints
- Copies packages into `modules/`
- Writes/updates `nano.lock`
- Recursively installs sub-dependencies

### `nanoc add <package>[@range] ...`

I also expose a compiler-level shortcut. `nanoc add` forwards to my installer with `--save`, so I record the dependency in `nano.packages.json`, install it, and update `nano.lock` in one step.

### `nanoc-pkg publish`

Package and stage the current directory for registry publication.

- Reads `[package]` from `nano.toml`
- Validates version format (semver)
- Copies `.nano`, `.c`, `.h`, `module.json`, `nano.toml`, `README.md`, `LICENSE`
- Commits to local registry clone
- You push when ready

### `nanoc-pkg update`

Re-resolve all dependencies to latest compatible versions, ignoring the lockfile.

### `nanoc-pkg init`

Create a new `nano.toml` with defaults based on the current directory name.

### `nanoc-pkg list`

Show installed packages from `nano.lock`.

### `nanoc-pkg info <package>`

Show registry metadata for a package.

## Integration with My Build System

The package manager installs modules into the standard `modules/` directory. My existing module system (`module_builder.c`) already knows how to find, build, and link modules from that directory. No special integration is needed for basic usage.

For build automation, the Makefile provides:

```bash
make pkg-install    # Equivalent to nanoc-pkg install
make pkg-publish    # Equivalent to nanoc-pkg publish
```

### Module Resolution Order

When I compile your code and encounter an `import`:

1. Current directory
2. `NANO_MODULE_PATH` directories (colon-separated)
3. `modules/` directory (where `nanoc-pkg install` puts packages)

Packages installed by `nanoc-pkg` land in `modules/` and are immediately available to the compiler without additional configuration.

### Build Cache Interaction

My module builder (`module_builder.c`) tracks content hashes for incremental builds. When `nanoc-pkg install` updates a module, the content hash changes, triggering a rebuild on next compilation. This is automatic.

I rebuild when the hash record is missing, unreadable, malformed or disagrees
with current inputs. Older or unchanged timestamps cannot override a content
change. I include regular and shared-only C sources, the manifest, and headers
recorded in compiler dependency files. Shared-only sources use the same selected
compiler as the ordinary module sources and produce dependency files too.

I hold a per-cache process lock through validation, compilation and publication.
I compile into a private directory, require nonempty regular artifacts, and
publish complete files by rename. Failed compilation leaves cached files intact.
I invalidate hash evidence before publication, so interrupted or failed
publication forces rebuilding. A killed build may leave a private directory;
later builds ignore it. I do not remove other invocations' directories.

I also compare a versioned build-context digest: the selected compiler command
(`NANO_CC`, then `CC`, then the manifest, then `cc`), its resolved executable
path and bytes, and the working directory. I include these environment inputs,
distinguishing unset from empty values:

- Search paths: `PATH`, `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`,
  `OBJC_INCLUDE_PATH`, `LIBRARY_PATH`, `COMPILER_PATH`, `GCC_EXEC_PREFIX`.
- SDK and flags: `SDKROOT`, `DEVELOPER_DIR`, `MACOSX_DEPLOYMENT_TARGET`,
  `IPHONEOS_DEPLOYMENT_TARGET`, `ARCHFLAGS`, `CFLAGS`, `CPPFLAGS`, `LDFLAGS`.
- Package configuration: `PKG_CONFIG_PATH`, `PKG_CONFIG_LIBDIR`,
  `PKG_CONFIG_SYSROOT_DIR`.
- Driver loading: `LD_LIBRARY_PATH`, `LD_PRELOAD`, `LD_AUDIT`,
  `DYLD_LIBRARY_PATH`, `DYLD_FALLBACK_LIBRARY_PATH`, `DYLD_INSERT_LIBRARIES`.
- Other build inputs: `SOURCE_DATE_EPOCH`, `LANG`, `LC_ALL`, `LC_CTYPE`,
  `NANO_TOOLCHAIN_ID`.

The digest does not make unused flags active. I persist the digest, not these
environment strings. Existing records without this context rebuild once.
Compiler shell expressions or unreadable/unresolved executables do not qualify
for cache reuse. I still attempt the requested build; on success I leave no
reusable hash record. If driver/context changes across a build, I likewise
withhold reuse evidence.

`NANO_TOOLCHAIN_ID` is a caller-supplied invalidation stamp for inputs I do not
discover, such as a wrapper's configuration or a managed SDK revision. Change
it when those inputs change. It is not a verified toolchain digest. Driver
hashing does not identify every compiler subprocess, linker, library,
pkg-config result or wrapper dependency; their automatic tracking remains open.

These hashes are a non-cryptographic cache optimization, not artifact
authentication. Publication is atomic per file, not across the artifact set;
later readers do not yet hold immutable artifact bindings. Power-loss durability,
source snapshots and complete toolchain identity remain open work. A matching
source hash alone does not establish a reproducible build.

### System Dependencies During Compilation

I check declared `pkg_config` dependencies on every module build, including
warm-cache builds. By default I report a missing dependency without running a
package manager, sudo, or package-registry install/probe commands. I also report
a missing `pkg-config` executable without installing it. A manifest without
`pkg_config` entries has no such availability probe; a later compiler or linker
error can still expose a missing dependency.

Install system dependencies separately, or opt in for one trusted build:

```bash
NANO_ALLOW_PACKAGE_INSTALL=1 ./bin/nanoc_c program.nano -o program
```

Only the exact value `1` enables my module builder's existing installation
paths. That opt-in permits package-registry commands and host package-manager
operations; it is not a promise that installation will succeed. It does not
affect explicit package-installation tools such as `nanoc-pkg`.

This default is not a build sandbox. Module build flags and foreign code still
require trust, and my separate shell-argument boundary audit remains open.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NANO_REGISTRY` | `https://github.com/jordanhubbard/nano-packages.git` | Registry Git URL |
| `NANO_REGISTRY_BRANCH` | `main` | Registry branch |
| `NANO_PKG_CACHE` | `~/.cache/nanolang/packages` | Local cache directory |
| `NANO_VERBOSE_BUILD` | `0` | Set to `1` for verbose output |
| `NANO_ALLOW_PACKAGE_INSTALL` | unset | Only `1` permits system-package installation by the compiler's module builder |

## Example

Here is a complete workflow:

```bash
# Create a new project
mkdir my_project && cd my_project
nanoc-pkg init

# Edit nano.toml to add dependencies
cat >> nano.toml << 'EOF'

[dependencies]
vector2d = ">=1.0.0"
EOF

# Install dependencies
nanoc-pkg install
#   ✓ vector2d@1.0.0

# Write your code
cat > main.nano << 'EOF'
import "modules/vector2d/vector2d.nano"

fn main() -> Int {
    let v: Vector2D = vector2d_new(3.0, 4.0)
    let len: Float = vector2d_length(v)
    print("Length: " + float_to_string(len))
    return 0
}
EOF

# Build
nanoc main.nano -o my_project

# Commit the lockfile
git add nano.toml nano.lock
git commit -m "Add vector2d dependency"
```

## Design Decisions

**Why Git-based?** Because Git is already everywhere. No new server infrastructure. Anyone can mirror or fork the registry. Package contents are versioned, diffable, and auditable.

**Why TOML?** Because it is the simplest format that handles nested sections and arrays cleanly. JSON requires too many quotes. YAML has too many footguns.

**Why shell scripts?** Because I do not want to add C dependencies for package management. The compiler stays lean. The tooling layer is the right place for network access, git operations, and file shuffling.

**Why not a lockfile in TOML?** Because the lockfile is machine-generated and machine-consumed. JSON is better for that. Humans should not edit it.
