# My Package Manager Design

**Status:** Design Proposal (Not Yet Implemented)  
**Version:** 1.0  
**Created:** January 25, 2026  

## Overview

This document outlines the design for nanopkg, my package manager. I have designed this to provide a simple, reliable way to share and use NanoLang libraries.

## Goals

### My Primary Goals
- Simple package installation (`nanopkg install <package>`)
- Dependency resolution (I will automatically install dependencies)
- Version management (I use semantic versioning)
- Integration with my existing module system
- Support for local and remote packages
- Reproducible builds via lock files

### My Secondary Goals
- Package publishing (`nanopkg publish`)
- Package search (`nanopkg search <term>`)
- Documentation hosting
- Automated testing of packages

### What I Will Not Do (For v1)
- Binary distribution (I compile from source only)
- Cross-compilation (I build on the target platform)
- Private registries (I support the public registry only)
- Workspaces (I do not support monorepos yet)

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
- Namespace prefixes: `nano-*` (I reserve these for official packages)

## Versioning

### Semantic Versioning

My packages use [SemVer 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH

1.2.3
│ │ │
│ │ └─ Patch: Bug fixes (backward compatible)
│ └─── Minor: New features (backward compatible)
└───── Major: Breaking changes
```

**Examples:**
- `1.2.3` to `1.2.4`: Bug fix, safe to update
- `1.2.3` to `1.3.0`: New feature, safe to update
- `1.2.3` to `2.0.0`: Breaking change, requires code changes

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

**Default:** Caret (`^`). I allow minor and patch updates by default.

### Pre-release Versions

```
1.0.0-alpha.1
1.0.0-beta.2
1.0.0-rc.1
1.0.0          # Stable release
```

I do not install pre-release versions unless you explicitly request them.

## Dependency Resolution

### My Algorithm

1. I parse package.json and its dependencies.
2. I fetch package metadata from my registry.
3. I resolve versions based on your constraints.
4. I check for conflicts.
5. I build a dependency graph.
6. I install packages in topological order.

### Conflict Resolution

**Scenario:** Two packages require different versions of the same dependency.

```
my-app
├── pkg-a@1.0.0
│   └── utils@^1.0.0  (requires 1.x)
└── pkg-b@1.0.0
    └── utils@^2.0.0  (requires 2.x)
```

**My Resolution Strategy:**

1. **Try to satisfy both.** I find a version that satisfies both constraints.
   - If possible: I use a single version.
   - Example: `^1.5.0` and `^1.2.0` leads me to use `1.5.0`.

2. **Fail if incompatible.** I cannot resolve major version conflicts.
   - Error: "Cannot resolve utils: pkg-a requires ^1.0, pkg-b requires ^2.0"
   - You must update one package or resolve the conflict manually.

**No Duplicate Versions (v1):** I do not allow multiple versions of the same package. I value simplicity over flexibility.

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

**What I do:**

1. I read your package.json if it exists.
2. I fetch package metadata from my registry.
3. I resolve your dependencies.
4. I download packages to `~/.nanopkg/cache/`.
5. I install them to your project's `nanopkg_modules/` directory.
6. I update your `package-lock.json`.
7. I compile the code if necessary.

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

**Why I use lock files:**
- Reproducible builds (I use the same versions every time).
- Faster installs (I do not need to resolve dependencies again).
- You should commit these to your version control.

## Registry

### My Package Registry

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

### My Registry Implementation

**Phase 1 (MVP):**
- Static file hosting without a database.
- Manual package approval.
- GitHub releases serve as the backend.

**Phase 2 (Full):**
- Database-backed registry.
- Automated publishing.
- CDN distribution.
- Package statistics.

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

I create a new `package.json` file interactively:

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
Validating package.json
Running tests
Building package
Creating tarball
Uploading to registry
Published json-parser@1.2.3
```

**My Pre-publish Checks:**
- I validate your package.json.
- I ensure all required files are present.
- I verify that your tests pass.
- I check for unpublished changes in git.
- I ensure the version is not already published.
- I verify the license file exists.

## Integration with My Module System

### My Current Module System

I already provide a module system:

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

I search for modules in this order:
1. The current directory.
2. The `./nanopkg_modules/` directory.
3. The `~/.nanopkg/global/` directory.
4. The path specified in `$NANO_MODULE_PATH`.

## Security

### Package Security

**Verification:**
- I use SHA-256 checksums for all packages.
- I verify integrity before installation.
- I only allow HTTPS connections to my registry.

**Sandboxing (Future):**
- I plan to add a package permissions system.
- I will restrict filesystem access.
- I will implement network access controls.
- I will provide audit logs.

**Publishing Security:**
- I require authentication for publishing.
- I will offer optional package signing in the future.
- I perform automated malware scanning.

## Caching

### My Local Cache

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

### My Global Config (~/.nanopkg/config.json)

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

## My Roadmap

### Phase 1: MVP (v0.1.0) - Q2 2026

- [ ] Basic package.json support
- [ ] Install command for local packages
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
- [ ] Pre and post hooks
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

### Q: Why did I not use an existing package manager?

A: I require optimizations specific to my design, a simpler model, and tight integration with my compiler.

### Q: How do you publish a package?

A: You use `nanopkg publish` after you have run `nanopkg login`. I require authentication.

### Q: Can you use private packages?

A: Not in v1. I have planned this for future versions.

### Q: How do you vendor dependencies?

A: You can copy `nanopkg_modules/` to your version control, although I do not recommend it. You can also use `nanopkg pack`.

### Q: What about binary packages?

A: I compile from source in v1. I have planned binary distribution for v2.

### Q: How do you contribute to my registry?

A: You should contact my maintainers or submit an RFC for registry contributions.

## Implementation Notes

**My Recommended Implementation Language:** NanoLang (self-hosted)  
**Alternative:** C or Go for bootstrap  
**Estimated Effort:** 4 to 6 weeks for MVP  
**Team Size:** 1 to 2 developers  

## References

- [npm Documentation](https://docs.npmjs.com/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Go Modules](https://go.dev/ref/mod)
- [SemVer 2.0.0](https://semver.org/)

---

**Status:** Design Proposal  
**My Next Steps:**
1. Gather community feedback through my RFC process.
2. Prototype my basic package.json parsing.
3. Implement my install command for local use.
4. Build my registry infrastructure.

**Created:** January 25, 2026  
**Author:** NanoLang Core Team
