# Release Process

This document describes the automated release process for NanoLang.

## Overview

NanoLang uses an automated release system that:
1. Bumps the version number (major, minor, or patch)
2. Generates CHANGELOG entry from git commits
3. Creates git tag
4. Commits and pushes changes
5. Creates GitHub release

## Quick Start

```bash
# Patch release (x.y.Z) - bug fixes
make release

# Minor release (x.Y.0) - new features, backward compatible
make release-minor

# Major release (X.0.0) - breaking changes
make release-major
```

### Batch Mode (Non-Interactive)

For CI/CD automation, use batch mode to skip all prompts:

```bash
# Non-interactive releases (requires clean state and passing tests)
BATCH=yes make release
BATCH=yes make release-minor
BATCH=yes make release-major
```

**Batch mode behavior:**
- Skips all confirmation prompts
- Fails fast on errors (wrong branch, dirty repo, test failures)
- Useful for CI/CD pipelines and automated releases

## Prerequisites

### Required Tools
- **Git**: For version control
- **GitHub CLI (`gh`)**: For creating releases
  ```bash
  brew install gh
  gh auth login
  ```

### Required State
- Working directory must be clean (no uncommitted changes)
- All tests should pass
- On `main` branch (or confirm to proceed from other branch)

## Detailed Steps

### 1. Prepare for Release

Before running the release command:

```bash
# Ensure all changes are committed
git status

# Run tests to verify everything works
make test

# Review recent commits
git log --oneline -10
```

### 2. Run Release Command

```bash
# For a patch release (most common)
make release

# The script will:
# 1. Check prerequisites (gh installed, git clean, etc.)
# 2. Detect current version from git tags
# 3. Calculate next version
# 4. Show you what will be released
# 5. Ask for confirmation
```

### 3. Review Generated Changelog

The script automatically generates a CHANGELOG entry from your git commits:

**Commit Message Format** (conventional commits):
- `feat:` or `feat(scope):` → Added section
- `fix:` or `fix(scope):` → Fixed section  
- `refactor:` or `refactor(scope):` → Changed section
- `chore:` or `docs:` → Other section

Example:
```bash
git commit -m "feat(stdlib): Add beads module"
git commit -m "fix(transpiler): Handle for-in loops"
git commit -m "docs: Update CHANGELOG"
```

### 4. Confirm and Release

The script will:
1. Show you the proposed version number
2. Display the generated changelog entry
3. Ask for confirmation twice (once for version, once for changelog)
4. Run tests
5. Update CHANGELOG.md
6. Create git tag
7. Commit changes
8. Push to GitHub
9. Create GitHub release

## Commit Message Best Practices

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `chore`: Maintenance tasks
- `test`: Test changes
- `perf`: Performance improvements

### Examples

```bash
# Good commit messages
git commit -m "feat: Add property-based testing framework"
git commit -m "fix(transpiler): Properly handle for-in loops"
git commit -m "refactor(stdlib): Simplify string operations"
git commit -m "docs: Add release process documentation"

# These will be categorized correctly in CHANGELOG
```

## Versioning

NanoLang follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes, incompatible API changes
- **MINOR** (x.Y.0): New features, backward compatible
- **PATCH** (x.y.Z): Bug fixes, backward compatible

### When to Bump

**Patch (x.y.Z):**
- Bug fixes
- Documentation updates
- Internal refactoring (no API changes)
- Performance improvements

**Minor (x.Y.0):**
- New features
- New APIs
- Deprecations (but not removals)
- Significant enhancements

**Major (X.0.0):**
- Breaking API changes
- Removed deprecated features
- Complete rewrites
- Incompatible changes

## Troubleshooting

### "GitHub CLI is not installed"
```bash
brew install gh
gh auth login
```

### "Git working directory is not clean"
```bash
# Commit your changes first
git add .
git commit -m "feat: Your changes"

# Or stash them
git stash
```

### "Tests failed"
```bash
# Fix the failing tests first
make test

# Or force release (not recommended)
# The script will ask if you want to continue despite failures
```

### "Not on main branch"
The script will warn you but allow you to proceed (interactive mode only). This is useful for:
- Release branches
- Hotfix branches
- Testing the release process

**Note:** In batch mode (`BATCH=yes`), the script will fail if not on main branch.

### Manual Release

If the automated script fails, you can release manually:

```bash
# 1. Update CHANGELOG.md manually
vim planning/CHANGELOG.md

# 2. Commit
git add planning/CHANGELOG.md
git commit -m "docs: Update CHANGELOG for v2.0.6"

# 3. Create tag
git tag -a v2.0.6 -m "Release v2.0.6"

# 4. Push
git push origin main
git push origin v2.0.6

# 5. Create GitHub release
gh release create v2.0.6 \
  --title "v2.0.6" \
  --notes "See CHANGELOG.md for details"
```

## Example Release Session

```bash
$ make release

╔═══════════════════════════════════════╗
║   NanoLang Automated Release Script   ║
╚═══════════════════════════════════════╝

ℹ️  Checking prerequisites...
✅ Prerequisites check passed
ℹ️  Current version: v2.0.5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Current: v2.0.5
  Next:    v2.0.6 (patch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Proceed with release v2.0.6? (y/n) y

ℹ️  Generating changelog from v2.0.5 to HEAD...

ℹ️  Generated changelog entry:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## [2.0.6] - 2026-01-14

### Fixed
- Properly handle edge case in parser

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Does this look correct? (y/n) y

ℹ️  Updating planning/CHANGELOG.md...
✅ CHANGELOG.md updated
ℹ️  Running tests...
✅ Tests passed
ℹ️  Creating release v2.0.6...
ℹ️  Creating git tag v2.0.6...
ℹ️  Committing CHANGELOG.md...
ℹ️  Pushing to origin...
ℹ️  Creating GitHub release...
✅ Release v2.0.6 created successfully!

╔═══════════════════════════════════════╗
║    🎉 Release Complete! 🎉            ║
╚═══════════════════════════════════════╝

Release: https://github.com/jordanhubbard/nanolang/releases/tag/v2.0.6
```

## Tips

1. **Commit frequently** with good messages during development
2. **Use conventional commit format** for automatic categorization
3. **Run tests** before releasing: `make test`
4. **Review git log** before releasing: `git log v2.0.5..HEAD --oneline`
5. **Test the release script** on a branch first if unsure

## See Also

- [CHANGELOG.md](../planning/CHANGELOG.md) - Version history
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
