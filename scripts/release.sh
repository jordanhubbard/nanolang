#!/bin/bash
# Automated release script for NanoLang
# Usage: ./scripts/release.sh [major|minor|patch]
# Batch mode: BATCH=yes ./scripts/release.sh [major|minor|patch]

set -euo pipefail

# Batch mode detection
BATCH_MODE="${BATCH:-no}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check if gh CLI is installed
    if ! command -v gh &> /dev/null; then
        error "GitHub CLI (gh) is not installed. Install with: brew install gh"
    fi
    
    # Check if gh is authenticated
    if ! gh auth status &> /dev/null; then
        error "GitHub CLI is not authenticated. Run: gh auth login"
    fi
    
    # Check if git repo is clean
    if [[ -n $(git status --porcelain) ]]; then
        error "Git working directory is not clean. Commit or stash changes first."
    fi
    
    # Check we're on main branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$CURRENT_BRANCH" != "main" ]]; then
        warn "Not on main branch (currently on: $CURRENT_BRANCH)"
        if [[ "$BATCH_MODE" == "yes" ]]; then
            error "Not on main branch in batch mode. Switch to main first."
        fi
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Aborted by user"
        fi
    fi
    
    success "Prerequisites check passed"
}

# Get current version from git tags
get_current_version() {
    git tag -l 'v*' | sort -V | tail -1 | sed 's/^v//'
}

# Calculate next version
calculate_next_version() {
    local current=$1
    local bump_type=$2
    
    # Parse current version
    IFS='.' read -r major minor patch <<< "$current"
    
    case $bump_type in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            error "Invalid bump type: $bump_type (use major, minor, or patch)"
            ;;
    esac
    
    echo "$major.$minor.$patch"
}

# Generate changelog entry from git log
generate_changelog_entry() {
    local prev_version=$1
    local new_version=$2
    local date=$(date +%Y-%m-%d)
    
    info "Generating changelog from v$prev_version to HEAD..." >&2

    # Get commits since last version (handle first-release case)
    local commits
    if git rev-parse "v$prev_version" &>/dev/null; then
        commits=$(git log "v$prev_version"..HEAD --pretty=format:"%h %s" --no-merges)
    else
        commits=$(git log --pretty=format:"%h %s" --no-merges)
    fi
    
    # Categorize commits
    local added=""
    local changed=""
    local fixed=""
    local removed=""
    local other=""
    
    while IFS= read -r line; do
        if [[ $line =~ ^[a-f0-9]+\ feat(\(.*\))?:\ (.*) ]]; then
            added+="- ${BASH_REMATCH[2]}\n"
        elif [[ $line =~ ^[a-f0-9]+\ fix(\(.*\))?:\ (.*) ]]; then
            fixed+="- ${BASH_REMATCH[2]}\n"
        elif [[ $line =~ ^[a-f0-9]+\ refactor(\(.*\))?:\ (.*) ]]; then
            changed+="- ${BASH_REMATCH[2]}\n"
        elif [[ $line =~ ^[a-f0-9]+\ (chore|docs)(\(.*\))?:\ (.*) ]]; then
            other+="- ${BASH_REMATCH[3]}\n"
        else
            # Extract commit message after hash
            local msg=$(echo "$line" | cut -d' ' -f2-)
            other+="- $msg\n"
        fi
    done <<< "$commits"
    
    # Build changelog entry
    local entry="## [$new_version] - $date\n\n"
    
    if [[ -n "$added" ]]; then
        entry+="### Added\n$added\n"
    fi
    
    if [[ -n "$changed" ]]; then
        entry+="### Changed\n$changed\n"
    fi
    
    if [[ -n "$fixed" ]]; then
        entry+="### Fixed\n$fixed\n"
    fi
    
    if [[ -n "$removed" ]]; then
        entry+="### Removed\n$removed\n"
    fi
    
    echo -e "$entry"
}

# Update CHANGELOG.md
update_changelog() {
    local changelog_entry=$1
    local changelog_file="CHANGELOG.md"

    info "Updating $changelog_file..."

    if [[ ! -f "$changelog_file" ]]; then
        error "CHANGELOG.md not found at $changelog_file"
    fi
    
    # Create temp files
    local temp_file=$(mktemp)
    local entry_file=$(mktemp)
    
    # Write the entry to a file (handles multi-line strings with emoji)
    echo -e "$changelog_entry" > "$entry_file"
    
    # Read changelog and insert new entry after ## [Unreleased]
    awk '
        /^## \[Unreleased\]/ {
            print $0
            print ""
            # Read and insert the new entry from file
            while ((getline line < "'"$entry_file"'") > 0) {
                print line
            }
            close("'"$entry_file"'")
            next
        }
        { print }
    ' "$changelog_file" > "$temp_file"
    
    mv "$temp_file" "$changelog_file"
    rm "$entry_file"
    
    success "CHANGELOG.md updated"
}

# Create git tag and release notes
create_release() {
    local version=$1
    local prev_version=$2
    local test_status=$3  # Passed from caller to avoid running tests twice
    
    info "Creating release v$version..."
    
    # Get changelog entry for release notes (handle first-release case)
    local release_notes commit_count
    if git rev-parse "v$prev_version" &>/dev/null; then
        release_notes=$(git log "v$prev_version"..HEAD --pretty=format:"- %s" --no-merges)
        commit_count=$(git rev-list --count "v$prev_version"..HEAD)
    else
        release_notes=$(git log --pretty=format:"- %s" --no-merges)
        commit_count=$(git rev-list --count HEAD)
    fi
    # test_status is now passed as argument (no longer runs make test again)
    
    # Build release notes
    local compare_url=""
    if [[ -n "${REPO_URL:-}" ]]; then
        compare_url="${REPO_URL}/compare/v${prev_version}...v${version}"
    fi

    cat > /tmp/release_notes.md << EOF
## NanoLang v$version

### Statistics
- **Commits since v$prev_version**: $commit_count
- **Test Status**: $test_status

### Changes

$release_notes
EOF
    if [[ -n "$compare_url" ]]; then
        printf '\n### Links\n- [Full Changelog](%s)\n- [Documentation](%s/tree/main/docs)\n\n---\n\n**Full Changelog**: %s\n' \
            "$compare_url" "${REPO_URL}" "$compare_url" >> /tmp/release_notes.md
    fi
    
    # Commit changelog (if there are changes to commit)
    info "Committing CHANGELOG.md..."
    git add CHANGELOG.md
    if git diff --cached --quiet; then
        info "CHANGELOG.md already up to date, skipping commit"
    else
        git commit -m "docs: Update CHANGELOG for v$version release

Release highlights from v$prev_version

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
    fi

    # Create annotated git tag (after changelog commit so tag includes it)
    info "Creating git tag v$version..."
    git tag -a "v$version" -m "Release v$version"
    
    # Push commits and tags
    info "Pushing to origin..."
    git push origin main
    git push origin "v$version"
    
    # Create GitHub release
    info "Creating GitHub release..."
    gh release create "v$version" \
        --title "v$version" \
        --notes-file /tmp/release_notes.md
    
    # Clean up
    rm /tmp/release_notes.md
    
    success "Release v$version created successfully!"
}

# Main script
main() {
    echo ""
    echo "╔═══════════════════════════════════════╗"
    echo "║   NanoLang Automated Release Script   ║"
    echo "╚═══════════════════════════════════════╝"
    echo ""
    
    if [[ "$BATCH_MODE" == "yes" ]]; then
        info "Running in BATCH mode (non-interactive)"
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Get current version
    CURRENT_VERSION=$(get_current_version)

    # Resolve repo URL dynamically (used in release notes)
    REPO_URL=$(gh repo view --json url -q .url 2>/dev/null || echo "")

    if [[ -z "$CURRENT_VERSION" ]]; then
        # First release — bootstrap from 1.0.0
        CURRENT_VERSION="0.0.0"
        info "No prior tags found — this will be the first release (v1.0.0)"
    else
        info "Current version: v$CURRENT_VERSION"
    fi
    
    # Determine bump type
    BUMP_TYPE=${1:-patch}
    if [[ ! "$BUMP_TYPE" =~ ^(major|minor|patch)$ ]]; then
        error "Invalid argument: $BUMP_TYPE (use major, minor, or patch)"
    fi
    
    # Calculate next version
    NEXT_VERSION=$(calculate_next_version "$CURRENT_VERSION" "$BUMP_TYPE")
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Current: v$CURRENT_VERSION"
    echo "  Next:    v$NEXT_VERSION ($BUMP_TYPE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Confirm
    if [[ "$BATCH_MODE" == "yes" ]]; then
        info "Batch mode: proceeding with release v$NEXT_VERSION"
    else
        read -p "Proceed with release v$NEXT_VERSION? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            warn "Release cancelled by user"
            exit 0
        fi
    fi
    
    # Generate changelog entry
    CHANGELOG_ENTRY=$(generate_changelog_entry "$CURRENT_VERSION" "$NEXT_VERSION")
    
    echo ""
    info "Generated changelog entry:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "$CHANGELOG_ENTRY"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [[ "$BATCH_MODE" == "yes" ]]; then
        info "Batch mode: accepting changelog entry"
    else
        read -p "Does this look correct? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            warn "Please edit planning/CHANGELOG.md manually and re-run"
            exit 0
        fi
    fi
    
    # Update changelog
    update_changelog "$CHANGELOG_ENTRY"
    
    # Run tests before release (capture output for release notes)
    info "Running tests..."
    local test_output_file=$(mktemp)
    if ! make test > "$test_output_file" 2>&1; then
        if [[ "$BATCH_MODE" == "yes" ]]; then
            rm -f "$test_output_file"
            error "Tests failed in batch mode. Fix tests before releasing."
        fi
        warn "Tests failed! Continue anyway?"
        read -p "(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$test_output_file"
            error "Release cancelled due to test failures"
        fi
    fi
    success "Tests passed"
    
    # Extract test status from captured output (avoid running tests twice)
    local test_status=$(grep -E "TOTAL:|passed|failed" "$test_output_file" | tail -1 || echo "All tests passed")
    rm -f "$test_output_file"
    
    # Create release
    create_release "$NEXT_VERSION" "$CURRENT_VERSION" "$test_status"
    
    echo ""
    echo "╔═══════════════════════════════════════╗"
    echo "║    🎉 Release Complete! 🎉            ║"
    echo "╚═══════════════════════════════════════╝"
    echo ""
    echo "Release: https://github.com/jordanhubbard/nanolang/releases/tag/v$NEXT_VERSION"
    echo ""
}

# Run main
main "$@"
