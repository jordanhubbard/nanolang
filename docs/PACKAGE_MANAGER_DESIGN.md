# NanoLang Package Manager Design

**Status:** Design Proposal (Not Yet Implemented)  
**Version:** 1.0  
**Created:** January 25, 2026  

## Overview

This document outlines the design for `nanopkg`, a package manager for NanoLang. The goal is to provide a simple, reliable way to share and use NanoLang libraries.

## Goals

### Primary Goals
- ✅ Simple package installation (`nanopkg install <package>`)
- ✅ Dependency resolution (automatically install dependencies)
- ✅ Version management (semantic versioning)
- ✅ Integration with existing module system
- ✅ Local and remote packages
- ✅ Reproducible builds (lock files)

### Secondary Goals
- ⏳ Package publishing (`nanopkg publish`)
- ⏳ Package search (`nanopkg search <term>`)
- ⏳ Documentation hosting
- ⏳ Automated testing of packages

### Non-Goals (For v1)
- ❌ Binary distribution (compile from source only)
- ❌ Cross-compilation (build on target platform)
- ❌ Private registries (public registry only)
- ❌ Workspaces (monorepo support)

## Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│                   User                          │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│              nanopkg CLI                         │
│  (install, search, publish, update, etc.)       │
└────────────┬────────────────────────────────────┘
             │
             ├──────────────┬──────────────┐
             ▼              ▼              ▼
┌──────────────────┐ ┌─────────────┐ ┌──────────┐
│ Local Registry   │ │   Resolver  │ │ Builder  │
│  (~/.nanopkg/)   │ │  (deps)     │ │ (compile)│
└──────────────────┘ └─────────────┘ └──────────┘
             │              │              │
             ▼              ▼              ▼
┌─────────────────────────────────────────────────┐
│           Remote Registry                        │
│         (registry.nanolang.org)                  │
└─────────────────────────────────────────────────┘
```

## Package Format

### Package Structure

```
my-package/
├── package.json           # Package metadata
├── README.md              # Documentation
├── LICENSE                # License file
├── src/
│   ├── lib.nano          # Main library code (exported)
│   ├── internal.nano     # Internal implementation (not exported)
│   └── utils.nano        # Utilities
├── tests/
│   ├── test_lib.nano     # Tests
│   └── test_utils.nano
└── examples/
    └── example.nano      # Usage examples
```

### package.json Format

```json
{
  "name": "my-package",
  "version": "1.2.3",
  "description": "A great NanoLang package",
  "license": "MIT",
  "authors": [
    {
      "name": "Alice Developer",
      "email": "alice@example.com"
    }
  ],
  "homepage": "https://github.com/user/my-package",
  "repository": {
    "type": "git",
    "url": "https://github.com/user/my-package.git"
  },
  "keywords": ["math", "algorithms", "utilities"],
  "main": "src/lib.nano",
  "dependencies": {
    "another-package": "^2.0.0",
    "utils-lib": "~1.5.0"
  },
  "devDependencies": {
    "test-framework": "^1.0.0"
  },
  "nanolang": {
    "version": ">=0.3.0",
    "features": ["unions", "generics"]
  },
  "scripts": {
    "test": "nanoc tests/test_lib.nano -o /tmp/test && /tmp/test",
    "build": "nanoc src/lib.nano -o lib.so"
  }
}
```

### Naming Conventions

**Package Names:**
- Lowercase alphanumeric + hyphens
- Max 50 characters
- Must start with letter
- Examples: `json-parser`, `math-utils`, `http-client`

**Forbidden Names:**
- Reserved keywords: `test`, `std`, `core`, `builtin`
- Namespace prefixes: `nano-*` (reserved for official packages)

## Versioning

### Semantic Versioning

NanoLang packages use [SemVer 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH

1.2.3
│ │ │
│ │ └─ Patch: Bug fixes (backward compatible)
│ └─── Minor: New features (backward compatible)
└───── Major: Breaking changes
```

**Examples:**
- `1.2.3` → `1.2.4`: Bug fix, safe to update
- `1.2.3` → `1.3.0`: New feature, safe to update
- `1.2.3` → `2.0.0`: Breaking change, requires code changes

### Version Constraints

```json
{
  "dependencies": {
    "exact": "1.2.3",           // Exact version
    "caret": "^1.2.3",          // >=1.2.3 <2.0.0 (default)
    "tilde": "~1.2.3",          // >=1.2.3 <1.3.0
    "wildcard": "1.2.*",        // >=1.2.0 <1.3.0
    "range": ">=1.2.0 <2.0.0",  // Explicit range
    "latest": "*"               // Any version (not recommended)
  }
}
```

**Default:** Caret (`^`) - allows minor and patch updates.

### Pre-release Versions

```
1.0.0-alpha.1
1.0.0-beta.2
1.0.0-rc.1
1.0.0          # Stable release
```

Pre-release versions are NOT installed unless explicitly requested.

## Dependency Resolution

### Algorithm

1. **Parse** package.json and dependencies
2. **Fetch** package metadata from registry
3. **Resolve** versions using constraints
4. **Check** for conflicts
5. **Build** dependency graph
6. **Install** packages in topological order

### Conflict Resolution

**Scenario:** Two packages require different versions of same dependency.

```
my-app
├── pkg-a@1.0.0
│   └── utils@^1.0.0  (requires 1.x)
└── pkg-b@1.0.0
    └── utils@^2.0.0  (requires 2.x)
```

**Resolution Strategy:**

1. **Try to satisfy both** - Find version that satisfies both constraints
   - If possible: Use single version
   - Example: `^1.5.0` and `^1.2.0` → use `1.5.0`

2. **Fail if incompatible** - Major version conflicts cannot be resolved
   - Error: "Cannot resolve utils: pkg-a requires ^1.0, pkg-b requires ^2.0"
   - User must update one package or resolve manually

**No Duplicate Versions (v1):** Unlike npm, we don't allow multiple versions of the same package. Simplicity over flexibility.

## Package Installation

### Installation Process

```bash
# Install single package
nanopkg install json-parser

# Install specific version
nanopkg install json-parser@1.2.3

# Install all dependencies from package.json
nanopkg install
```

**What happens:**

1. ✅ Read package.json (if exists)
2. ✅ Fetch package metadata from registry
3. ✅ Resolve dependencies
4. ✅ Download packages to `~/.nanopkg/cache/`
5. ✅ Install to project `nanopkg_modules/`
6. ✅ Update `package-lock.json`
7. ✅ Compile if needed

### Directory Structure After Install

```
my-project/
├── package.json
├── package-lock.json          # Lock file (exact versions)
├── nanopkg_modules/           # Installed packages
│   ├── json-parser@1.2.3/
│   │   ├── package.json
│   │   └── src/
│   │       └── lib.nano
│   └── utils@2.1.0/
│       ├── package.json
│       └── src/
│           └── lib.nano
└── src/
    └── main.nano
```

### Lock Files (package-lock.json)

```json
{
  "lockfileVersion": 1,
  "dependencies": {
    "json-parser": {
      "version": "1.2.3",
      "resolved": "https://registry.nanolang.org/json-parser/1.2.3.tar.gz",
      "integrity": "sha256-abcd1234...",
      "dependencies": {
        "utils": "2.1.0"
      }
    },
    "utils": {
      "version": "2.1.0",
      "resolved": "https://registry.nanolang.org/utils/2.1.0.tar.gz",
      "integrity": "sha256-efgh5678..."
    }
  }
}
```

**Purpose:**
- Reproducible builds (same versions every time)
- Faster installs (no resolution needed)
- Commit to version control

## Registry

### Package Registry

**URL:** `https://registry.nanolang.org`

**API Endpoints:**

```
GET  /packages                    # List all packages
GET  /packages/{name}             # Get package metadata
GET  /packages/{name}/{version}   # Get specific version
GET  /packages/{name}/versions    # List all versions
POST /packages                    # Publish new package (authenticated)
GET  /search?q={query}            # Search packages
```

**Package Metadata Response:**

```json
{
  "name": "json-parser",
  "description": "JSON parsing library",
  "latest": "1.2.3",
  "versions": {
    "1.2.3": {
      "version": "1.2.3",
      "published": "2026-01-15T10:30:00Z",
      "tarball": "https://registry.nanolang.org/json-parser/1.2.3.tar.gz",
      "integrity": "sha256-abcd1234...",
      "dependencies": {
        "utils": "^2.0.0"
      }
    },
    "1.2.2": { /* ... */ },
    "1.2.1": { /* ... */ }
  },
  "keywords": ["json", "parser", "data"],
  "homepage": "https://github.com/user/json-parser",
  "license": "MIT"
}
```

### Registry Implementation

**Phase 1 (MVP):**
- Static file hosting (no database)
- Manual package approval
- GitHub releases as backend

**Phase 2 (Full):**
- Database-backed registry
- Automated publishing
- CDN distribution
- Package statistics

## CLI Commands

### Core Commands

```bash
# Initialize new package
nanopkg init

# Install dependencies
nanopkg install [package][@version]

# Update dependencies
nanopkg update [package]

# Remove package
nanopkg uninstall <package>

# List installed packages
nanopkg list

# Show package info
nanopkg info <package>

# Search packages
nanopkg search <query>

# Publish package
nanopkg publish

# Login to registry
nanopkg login

# Run script from package.json
nanopkg run <script>
```

### Command Details

#### `nanopkg init`

Creates new `package.json` interactively:

```bash
$ nanopkg init
Package name: my-app
Version: (1.0.0)
Description: My awesome app
Author: Alice Developer
License: (MIT)
Entry point: (src/main.nano)

Created package.json
```

#### `nanopkg install`

```bash
# Install all dependencies
nanopkg install

# Install specific package
nanopkg install json-parser

# Install dev dependency
nanopkg install --dev test-framework

# Install exact version
nanopkg install json-parser@1.2.3
```

#### `nanopkg publish`

```bash
$ nanopkg publish
✓ Validating package.json
✓ Running tests
✓ Building package
✓ Creating tarball
✓ Uploading to registry
✓ Published json-parser@1.2.3
```

**Pre-publish Checks:**
- ✅ Valid package.json
- ✅ All required files present
- ✅ Tests pass
- ✅ No unpublished changes (git clean)
- ✅ Version not already published
- ✅ License file exists

## Integration with Module System

### Current Module System

NanoLang already has a module system:

```nano
// Import from local file
from "utils.nano" import helper_function

// Import from module path
from "mymodule/lib.nano" import some_function
```

### Package Integration

```nano
// Import from installed package
from "nanopkg:json-parser" import parse_json, stringify

// Or use package path
from "nanopkg_modules/json-parser/src/lib.nano" import parse_json
```

**Environment Variable:**

```bash
export NANO_PACKAGE_PATH="./nanopkg_modules:~/.nanopkg/global"
```

Compiler searches:
1. Current directory
2. `./nanopkg_modules/`
3. `~/.nanopkg/global/`
4. `$NANO_MODULE_PATH`

## Security

### Package Security

**Verification:**
- ✅ SHA-256 checksums for all packages
- ✅ Verify integrity before installation
- ✅ HTTPS-only registry connections

**Sandboxing (Future):**
- ⏳ Package permissions system
- ⏳ Restrict filesystem access
- ⏳ Network access controls
- ⏳ Audit logs

**Publishing Security:**
- ✅ Authentication required for publishing
- ✅ Package signing (optional for v1)
- ✅ Malware scanning (automated)

## Caching

### Local Cache

```
~/.nanopkg/
├── cache/                 # Downloaded tarballs
│   ├── json-parser-1.2.3.tar.gz
│   └── utils-2.1.0.tar.gz
├── global/                # Global installations
│   └── json-parser@1.2.3/
└── config.json           # User configuration
```

**Cache Management:**

```bash
# Clear cache
nanopkg cache clean

# Show cache size
nanopkg cache size

# Verify cache integrity
nanopkg cache verify
```

## Configuration

### Global Config (~/.nanopkg/config.json)

```json
{
  "registry": "https://registry.nanolang.org",
  "cache": "~/.nanopkg/cache",
  "proxy": null,
  "auth": {
    "registry.nanolang.org": "token-here"
  },
  "defaults": {
    "license": "MIT",
    "author": "Alice Developer <alice@example.com>"
  }
}
```

### Project Config (.nanopkgrc)

```json
{
  "registry": "https://custom-registry.com",
  "installPath": "packages"
}
```

## Roadmap

### Phase 1: MVP (v0.1.0) - Q2 2026

- [ ] Basic package.json support
- [ ] Install command (local packages)
- [ ] Dependency resolution
- [ ] Lock file generation
- [ ] Integration with nanoc

### Phase 2: Registry (v0.2.0) - Q3 2026

- [ ] Remote registry implementation
- [ ] Publish command
- [ ] Search command
- [ ] Package metadata API
- [ ] Authentication

### Phase 3: Advanced Features (v0.3.0) - Q4 2026

- [ ] Update command with smart resolution
- [ ] Global installations
- [ ] Scripts support
- [ ] Pre/post hooks
- [ ] Package audit

### Phase 4: Ecosystem (v1.0.0) - 2027

- [ ] Documentation hosting
- [ ] Package discovery UI
- [ ] Statistics and analytics
- [ ] Private registries
- [ ] Automated testing

## Comparison with Other Package Managers

| Feature | nanopkg | npm | cargo | go modules |
|---------|---------|-----|-------|------------|
| **Versioning** | SemVer | SemVer | SemVer | Minimal |
| **Lock files** | Yes | Yes | Yes | Yes |
| **Workspaces** | No (v1) | Yes | Yes | Yes |
| **Multiple versions** | No | Yes | No | Yes |
| **Central registry** | Yes | Yes | Yes | Decentralized |
| **Complexity** | Low | High | Medium | Low |

## FAQ

### Q: Why not use existing package manager?

A: NanoLang-specific optimizations, simpler model, tight compiler integration.

### Q: How do I publish a package?

A: `nanopkg publish` after `nanopkg login`. Requires authentication.

### Q: Can I use private packages?

A: Not in v1. Planned for future versions.

### Q: How do I vendor dependencies?

A: Copy `nanopkg_modules/` to version control (not recommended) or use `nanopkg pack`.

### Q: What about binary packages?

A: v1 compiles from source. Binary distribution planned for v2.

### Q: How do I contribute to the registry?

A: Contact maintainers or submit RFC for registry contributions.

## Implementation Notes

**Recommended Implementation Language:** NanoLang (self-hosted)  
**Alternative:** C or Go for bootstrap  
**Estimated Effort:** 4-6 weeks for MVP  
**Team Size:** 1-2 developers  

## References

- [npm Documentation](https://docs.npmjs.com/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Go Modules](https://go.dev/ref/mod)
- [SemVer 2.0.0](https://semver.org/)

---

**Status:** Design Proposal  
**Next Steps:**
1. Gather community feedback (RFC process)
2. Prototype basic package.json parsing
3. Implement install command (local only)
4. Build registry infrastructure

**Created:** January 25, 2026  
**Author:** NanoLang Core Team
