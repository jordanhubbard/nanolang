#!/bin/bash
#
# Profile key NanoLang programs on Linux and collect optimization data
# Run this on ubuntu.local or any Linux system with gprofng
#
# Usage:
#   ./scripts/profile_tests.sh [output_dir]
#
# Output:
#   - Profiling JSON for each test in output_dir/
#   - Summary report with top hotspots
#   - Ready for LLM analysis

set -e

OUTPUT_DIR="${1:-build/profiling_results}"
PROFILE_FLAGS="-pg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== NanoLang Profiling Test Suite ===${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for gprofng (Linux profiling tool)
if ! command -v gprofng &> /dev/null; then
    echo -e "${YELLOW}Warning: gprofng not found. Profiling may not work.${NC}"
    echo "Install with: sudo apt install binutils (Ubuntu/Debian)"
    echo "              sudo dnf install binutils (Fedora)"
    echo ""
fi

# Ensure compiler is built
if [ ! -f "./bin/nanoc" ]; then
    echo -e "${RED}Error: ./bin/nanoc not found. Run 'make' first.${NC}"
    exit 1
fi

# Test programs to profile
declare -A TESTS=(
    ["test_std_collections"]="tests/test_std_collections.nano"
    ["test_hashmap_set"]="tests/test_hashmap_set_advanced.nano"
    ["userguide_build"]="scripts/userguide_build_html.nano"
    ["userguide_check"]="scripts/userguide_snippets_check.nano"
    ["syntax_highlighter"]="modules/nano_tools/nano_tools.nano"
)

# Statistics
TOTAL_TESTS=${#TESTS[@]}
SUCCESSFUL=0
FAILED=0

echo -e "${BLUE}Compiling and profiling $TOTAL_TESTS programs...${NC}"
echo ""

for test_name in "${!TESTS[@]}"; do
    test_file="${TESTS[$test_name]}"

    echo -e "${YELLOW}>>> Profiling: $test_name${NC}"
    echo "    Source: $test_file"

    # Output paths
    BINARY="$OUTPUT_DIR/${test_name}_profiled"
    PROFILE_JSON="$OUTPUT_DIR/${test_name}_profile.json"
    COMPILE_LOG="$OUTPUT_DIR/${test_name}_compile.log"
    RUN_LOG="$OUTPUT_DIR/${test_name}_run.log"

    # Compile with profiling
    echo -n "    Compiling... "
    if ./bin/nanoc "$test_file" -o "$BINARY" $PROFILE_FLAGS > "$COMPILE_LOG" 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "    See: $COMPILE_LOG"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Run and capture profiling output
    echo -n "    Running... "

    # Different execution for different test types
    case "$test_name" in
        "userguide_build")
            # Don't actually build userguide (would take too long)
            # Just profile the startup and initialization
            timeout 10s "$BINARY" --help > "$RUN_LOG" 2> "$PROFILE_JSON" || true
            ;;
        "syntax_highlighter")
            # Profile highlighting a sample file
            if [ -f "examples/language/nl_hello.nano" ]; then
                timeout 30s "$BINARY" examples/language/nl_hello.nano > "$RUN_LOG" 2> "$PROFILE_JSON" || true
            else
                echo "test" | timeout 10s "$BINARY" > "$RUN_LOG" 2> "$PROFILE_JSON" || true
            fi
            ;;
        *)
            # Regular test programs
            timeout 60s "$BINARY" > "$RUN_LOG" 2> "$PROFILE_JSON" || true
            ;;
    esac

    if [ -s "$PROFILE_JSON" ]; then
        echo -e "${GREEN}OK${NC}"
        SUCCESSFUL=$((SUCCESSFUL + 1))

        # Show top hotspot
        if command -v jq &> /dev/null; then
            TOP_HOTSPOT=$(jq -r '.hotspots[0] | "\(.function): \(.pct_time)%"' "$PROFILE_JSON" 2>/dev/null || echo "N/A")
            echo "    Top hotspot: $TOP_HOTSPOT"
        fi
    else
        echo -e "${YELLOW}NO DATA${NC}"
        echo "    Profile output empty - see: $RUN_LOG"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# Generate summary report
SUMMARY_FILE="$OUTPUT_DIR/summary.md"

echo -e "${BLUE}Generating summary report...${NC}"

cat > "$SUMMARY_FILE" << EOF
# NanoLang Profiling Results

**Date:** $(date)
**System:** $(uname -a)
**Tests Profiled:** $SUCCESSFUL/$TOTAL_TESTS successful

## Profiling Data Files

EOF

for test_name in "${!TESTS[@]}"; do
    PROFILE_JSON="$OUTPUT_DIR/${test_name}_profile.json"
    if [ -f "$PROFILE_JSON" ] && [ -s "$PROFILE_JSON" ]; then
        echo "- [\`${test_name}_profile.json\`](./${test_name}_profile.json)" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" << EOF

## Top Hotspots by Test

EOF

for test_name in "${!TESTS[@]}"; do
    PROFILE_JSON="$OUTPUT_DIR/${test_name}_profile.json"
    if [ -f "$PROFILE_JSON" ] && [ -s "$PROFILE_JSON" ]; then
        echo "### $test_name" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"

        if command -v jq &> /dev/null; then
            echo "\`\`\`json" >> "$SUMMARY_FILE"
            jq '.hotspots[:5]' "$PROFILE_JSON" 2>/dev/null >> "$SUMMARY_FILE" || echo "Error parsing JSON"
            echo "\`\`\`" >> "$SUMMARY_FILE"
        else
            echo "Install jq to see formatted results" >> "$SUMMARY_FILE"
        fi

        echo "" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" << EOF

## Next Steps

1. **Review profiling data** - Examine JSON files for performance bottlenecks
2. **LLM Analysis** - Feed profile data + source code to Claude/GPT for optimization suggestions
3. **Implement optimizations** - Apply suggested improvements
4. **Re-profile** - Run this script again to verify improvements

## LLM Prompt Template

\`\`\`
I'm profiling NanoLang programs for performance optimization. Here's the profiling data:

[paste JSON from one of the *_profile.json files]

And here's the relevant source code:

[paste source from the hot functions]

What optimizations do you recommend based on this profiling data?
\`\`\`

## Uploading to Main System

\`\`\`bash
# From ubuntu.local, copy results to main system
scp -r $OUTPUT_DIR/ your-mac:~/Downloads/profiling_results/

# Or commit to git
git add $OUTPUT_DIR/
git commit -m "profiling: results from ubuntu.local"
git push
\`\`\`

---

Generated by: scripts/profile_tests.sh
EOF

echo -e "${GREEN}Summary report: $SUMMARY_FILE${NC}"
echo ""

# Final statistics
echo -e "${BLUE}=== Results ===${NC}"
echo -e "Successful: ${GREEN}$SUCCESSFUL${NC}"
echo -e "Failed:     ${RED}$FAILED${NC}"
echo ""

if [ $SUCCESSFUL -gt 0 ]; then
    echo -e "${GREEN}Profiling data ready for LLM analysis!${NC}"
    echo "Next steps:"
    echo "  1. Review: cat $SUMMARY_FILE"
    echo "  2. Examine JSON files in: $OUTPUT_DIR/"
    echo "  3. Feed data to LLM for optimization suggestions"
else
    echo -e "${RED}No profiling data collected. Check logs in: $OUTPUT_DIR/${NC}"
    exit 1
fi
