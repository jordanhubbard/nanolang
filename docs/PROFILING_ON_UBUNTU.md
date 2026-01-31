# Profiling NanoLang on ubuntu.local

This guide shows how to run comprehensive profiling tests on ubuntu.local (Linux) where we can get actual profiling data using `gprofng`.

## Why Ubuntu?

**macOS Limitation:** System Integrity Protection (SIP) blocks the `sample` command from working properly, even with sudo. This makes automated profiling difficult on macOS.

**Linux Advantage:** `gprofng` (from binutils 2.39+) works without special permissions and provides excellent profiling data.

## Quick Start

### On ubuntu.local

```bash
# 1. Sync code to ubuntu.local
git pull  # Ensure latest code

# 2. Build compiler
make clean
make

# 3. Run profiling test suite
./scripts/profile_tests.sh

# 4. View results
cat build/profiling_results/summary.md
ls -lh build/profiling_results/*.json
```

### Copy Results Back to macOS

```bash
# From ubuntu.local
scp -r build/profiling_results/ your-mac:~/Downloads/

# Or commit to git
git add build/profiling_results/
git commit -m "profiling: results from ubuntu.local $(date +%Y-%m-%d)"
git push
```

## What Gets Profiled

The script profiles these key programs:

| Test | Purpose | What It Measures |
|------|---------|------------------|
| `test_std_collections` | Standard library collections | HashMap, Set, StringBuilder performance |
| `test_hashmap_set_advanced` | Advanced collection operations | Complex data structure usage |
| `userguide_build` | Documentation build | Markdown processing, syntax highlighting |
| `userguide_check` | Snippet validation | Parser performance on many files |
| `syntax_highlighter` | Pretty printing | Code highlighting speed |

## Output Files

After running, you'll have:

```
build/profiling_results/
├── summary.md                              # Overview report
├── test_std_collections_profile.json       # Profiling data
├── test_std_collections_profiled           # Binary
├── test_std_collections_compile.log        # Compilation output
├── test_std_collections_run.log            # Execution output
├── test_hashmap_set_profile.json
├── userguide_build_profile.json
├── userguide_check_profile.json
└── syntax_highlighter_profile.json
```

## Analyzing Results

### 1. Quick Look at Top Hotspots

```bash
# Show top 3 hotspots from each profile
for f in build/profiling_results/*_profile.json; do
    echo "=== $f ==="
    jq '.hotspots[:3]' "$f"
    echo ""
done
```

### 2. Find Functions Taking >10% Time

```bash
# Find expensive functions across all profiles
jq -r '.hotspots[] | select(.pct_time > 10.0) | "\(.function): \(.pct_time)%"' \
    build/profiling_results/*_profile.json | sort -t: -k2 -nr
```

### 3. LLM Analysis

Feed the data to an LLM for optimization suggestions:

**Example Prompt:**
```
I'm profiling NanoLang's standard library tests. Here's the profiling data showing
where time is spent:

[paste content of test_std_collections_profile.json]

And here's the source code for the top hotspot:

[paste relevant source from stdlib/]

What optimizations do you recommend? Consider:
- Algorithm complexity
- Memory allocations
- Cache efficiency
- Opportunities for batching operations
```

## Expected Hotspots

Based on the code, we expect to see:

### test_std_collections
- `hashmap_hash` - Hash function (should be ~20-40% of time)
- `hashmap_put` / `hashmap_get` - Core operations
- `stringbuilder_append` - String building
- `set_contains` - Set lookups

### userguide_build
- `pretty_print_html` - Syntax highlighting (optimized, should be ~30-40%)
- `md_to_html` - Markdown parsing
- `str_substring` - String operations
- `parse_snippet` - Code extraction

### syntax_highlighter
- `highlight_token` - Token colorization
- `str_matches_at` - Pattern matching (already optimized)
- `ansi_escape_for_token` - Color code generation

## Optimization Workflow

1. **Profile** → Run profiling script
2. **Identify** → Find functions taking >10% of time
3. **Analyze** → Feed profiling data + source to LLM
4. **Implement** → Apply suggested optimizations
5. **Verify** → Re-profile to confirm improvements
6. **Document** → Record optimizations in commit message

## Troubleshooting

### gprofng not found

```bash
# Install binutils (includes gprofng)
sudo apt install binutils       # Ubuntu/Debian
sudo dnf install binutils       # Fedora
```

### No profiling data in JSON files

**Symptom:** JSON files are empty or missing

**Possible causes:**
1. Program crashed before profiling could complete
2. Program ran too quickly (< 1 second)
3. gprofng not properly installed

**Solutions:**
```bash
# Check compilation logs
cat build/profiling_results/*_compile.log

# Check runtime logs
cat build/profiling_results/*_run.log

# Verify gprofng is working
gprofng --version

# Test manually
./bin/nanoc tests/test_std_collections.nano -o /tmp/test -pg
/tmp/test 2> /tmp/profile.json
cat /tmp/profile.json
```

### Timeout during profiling

**Symptom:** Tests timeout after 60 seconds

**Solution:**
Edit `scripts/profile_tests.sh` and increase timeout:
```bash
# Change from 60s to 180s
timeout 180s "$BINARY" > "$RUN_LOG" 2> "$PROFILE_JSON" || true
```

## Remote Execution from macOS

You can trigger profiling from your Mac:

```bash
# SSH to ubuntu.local and run profiling
ssh ubuntu.local 'cd ~/nanolang && git pull && make && ./scripts/profile_tests.sh'

# Copy results back
scp -r ubuntu.local:~/nanolang/build/profiling_results/ ./build/

# Analyze locally
cat build/profiling_results/summary.md
```

## Automated Profiling

Add to cron for regular profiling:

```bash
# On ubuntu.local, add to crontab
crontab -e

# Add line:
0 2 * * * cd ~/nanolang && git pull && make && ./scripts/profile_tests.sh ~/profiling_archive/$(date +\%Y\%m\%d)
```

This runs profiling nightly at 2am and archives results by date.

## CI Integration

We can also add this to GitHub Actions:

```yaml
name: Profile on Linux

on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2am

jobs:
  profile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: sudo apt-get update && sudo apt-get install -y binutils

      - name: Build compiler
        run: make

      - name: Run profiling tests
        run: ./scripts/profile_tests.sh

      - name: Upload profiling results
        uses: actions/upload-artifact@v3
        with:
          name: profiling-results
          path: build/profiling_results/

      - name: Comment on latest commit
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const summary = fs.readFileSync('build/profiling_results/summary.md', 'utf8');
            github.rest.repos.createCommitComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              commit_sha: context.sha,
              body: summary
            });
```

## Performance Baselines

Once we have initial profiling data, we can establish baselines:

**Target Metrics:**
- `test_std_collections`: < 2 seconds runtime
- `userguide_build`: < 30 seconds for full build
- `syntax_highlighter`: < 50ms per file

**Regression Detection:**
If profiling shows functions taking >2x expected time, investigate immediately.

## See Also

- **[userguide/08_profiling.md](../userguide/08_profiling.md)** - User guide chapter on profiling
- **[docs/PERFORMANCE.md](PERFORMANCE.md)** - General performance guide
- **[planning/AGENTS.md](../planning/AGENTS.md)** - Layer 4 profiling documentation

---

**Last Updated:** January 31, 2026
**Tested On:** Ubuntu 22.04 LTS with binutils 2.38+
