#!/bin/bash
# Automated release script for NanoLang
# Usage: ./scripts/release.sh [major|minor|patch]

set -euo pipefail

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
        error "Not on main branch (currently on: $CURRENT_BRANCH). Switch to main first."
    fi
    
    success "Prerequisites check passed"
}

# Check release-scoped GitHub work before changing release metadata. The full
# triage procedure lives in skills/release-readiness/SKILL.md.
check_release_github_work() {
    local version=$1
    local repo
    repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
    local scope="release/v$version"
    local milestone="v$version"

    info "Checking open GitHub work scoped to v$version..."
    local open_items=""
    while IFS= read -r item; do
        open_items+="$item\n"
    done < <(gh issue list --repo "$repo" --state open --limit 1000 \
        --json number,title,body,url,labels,milestone \
        | jq -r --arg scope "$scope" --arg milestone "$milestone" \
        '.[] | select((([.labels[].name] | index($scope)) or (.milestone.title == $milestone) or ((.title + " " + (.body // "")) | test($milestone)))) | "issue #\(.number): \(.title)\n\(.url)"')
    while IFS= read -r item; do
        open_items+="$item\n"
    done < <(gh pr list --repo "$repo" --state open --limit 1000 \
        --json number,title,body,url,labels,milestone \
        | jq -r --arg scope "$scope" --arg milestone "$milestone" \
        '.[] | select((([.labels[].name] | index($scope)) or (.milestone.title == $milestone) or ((.title + " " + (.body // "")) | test($milestone)))) | "PR #\(.number): \(.title)\n\(.url)"')

    if [[ -n "$open_items" ]]; then
        printf '%b' "$open_items" >&2
        error "Release-scoped GitHub issues or pull requests remain open. Triage them with skills/release-readiness/SKILL.md."
    fi
    success "No release-scoped GitHub work remains open"
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
    
    # Categorize commits. Every commit reaches a section: a changelog that
    # silently drops one is worse than a vague entry, because the omission is
    # invisible in the released artifact.
    local added=""
    local changed=""
    local fixed=""

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        if [[ $line =~ ^[a-f0-9]+\ feat(\(.*\))?:\ (.*) ]]; then
            added+="- ${BASH_REMATCH[2]}\n"
        elif [[ $line =~ ^[a-f0-9]+\ fix(\(.*\))?:\ (.*) ]]; then
            fixed+="- ${BASH_REMATCH[2]}\n"
        elif [[ $line =~ ^[a-f0-9]+\ (refactor|perf|test|chore|docs|build|ci|style)(\(.*\))?:\ (.*) ]]; then
            changed+="- ${BASH_REMATCH[3]}\n"
        else
            changed+="- $(echo "$line" | cut -d' ' -f2-)\n"
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

# Refresh root npm metadata for GitHub's dependency graph. The VS Code
# extension owns the JavaScript dependencies; the release owns the version.
update_package_json() {
    local version=$1
    info "Updating package.json..."
    python3 scripts/generate_root_package_json.py "$version"
    success "package.json updated"
}

# Create git tag and release notes
create_release() {
    local version=$1
    local prev_version=$2
    local test_status=$3  # Passed from caller to avoid running tests twice
    local release_branch="release/v$version"
    
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
    
    # Commit release metadata (if there are changes to commit)
    info "Committing release metadata..."
    git add CHANGELOG.md package.json
    if git diff --cached --quiet; then
        info "Release metadata already up to date, skipping commit"
    else
        git commit -m "docs: Update CHANGELOG for v$version release

Release highlights from v$prev_version

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
    fi

    # Rebase on any commits that landed on origin since we started, then put
    # the release commit on a branch so protected main is only changed by PR.
    info "Syncing with origin before creating the release PR..."
    git pull --rebase origin main
    git switch -c "$release_branch"

    info "Pushing $release_branch..."
    git push --set-upstream origin "$release_branch"

    info "Opening release PR..."
    local pr_url
    pr_url=$(gh pr create \
        --base main \
        --head "$release_branch" \
        --title "Release v$version" \
        --body "Prepare the v$version release.")

    info "Waiting for release PR checks..."
    gh pr checks "$pr_url" --watch --fail-fast

    info "Merging release PR..."
    gh pr merge "$pr_url" --squash --delete-branch

    # The protected branch may use squash or merge commits. Tag the commit
    # that actually landed instead of the now-obsolete release-branch commit.
    git switch main
    git pull --ff-only origin main
    info "Creating git tag v$version..."
    git tag -a "v$version" -m "Release v$version"
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
    check_release_github_work "$NEXT_VERSION"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Current: v$CURRENT_VERSION"
    echo "  Next:    v$NEXT_VERSION ($BUMP_TYPE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Generate changelog entry
    CHANGELOG_ENTRY=$(generate_changelog_entry "$CURRENT_VERSION" "$NEXT_VERSION")
    
    echo ""
    info "Generated changelog entry:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "$CHANGELOG_ENTRY"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Update changelog
    update_changelog "$CHANGELOG_ENTRY"

    # Refresh root dependency metadata using this release's exact version.
    update_package_json "$NEXT_VERSION"
    
    # Run tests before release (capture output for release notes)
    info "Running tests..."
    local test_output_file=$(mktemp)
    if ! make test > "$test_output_file" 2>&1; then
        rm -f "$test_output_file"
        error "Tests failed. Fix tests before releasing."
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
