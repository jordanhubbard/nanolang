# Advanced NanoLang Examples - Practical Problem Solving

## Executive Summary

This document outlines a comprehensive plan to create practical examples that demonstrate how NanoLang's advanced features (map/filter/fold, generics, AST manipulation) solve **real-world problems**. Current examples show syntax but not application. This plan addresses that gap with pedagogically-sound, industry-relevant examples.

## Current State Analysis

### Existing Examples
- **`nl_filter_map_fold.nano`** - Demonstrates mechanics (count_matching, apply_first, fold) but uses artificial data (arrays of integers)
- **`nl_generics_demo.nano`** - Shows List<T> syntax but artificial use cases (Point, Player structs without real purpose)
- **`stdlib_ast_demo.nano`** - Demonstrates AST API (ast_int, ast_string, ast_call) but no practical transformation
- **`nl_data_analytics.nano`** - Has potential but needs enhancement with real data pipelines

### The Gap
**Problem:** Examples demonstrate SYNTAX but not HOW to solve real-world problems.  
**Impact:** Developers can't see how to apply these features to their work.  
**Solution:** Create problem-first examples that start with a relatable challenge and show the solution.

## Proposed Examples (Priority Order)

### 1. Word Frequency Counter (`nl_word_frequency.nano`) ⭐ TOP PRIORITY
**Status:** 90% complete (in `/examples/nl_word_frequency.nano`)

**Problem Statement:**  
Given text input, count how many times each word appears and identify the most common words. This is fundamental to search engines, log analysis, and NLP.

**What It Demonstrates:**
- Map/filter/fold pipeline solving a concrete problem
- String processing (split on whitespace, normalize case, filter stopwords)
- Data transformation stages: text → words → normalized → filtered → counted → sorted
- Real-world applications: TF-IDF scoring, error pattern detection, keyword extraction

**Pipeline Stages:**
```
Input: "the quick brown fox jumps over the lazy dog"
  ↓ split_into_words (map)
["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
  ↓ normalize_word (map: lowercase, remove punctuation)
["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
  ↓ filter stopwords (filter: remove "the", "a", "is", etc.)
["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
  ↓ count frequencies (fold: accumulate counts)
[("quick", 1), ("brown", 1), ("fox", 1), ...]
  ↓ sort by frequency (sort)
  ↓ take top N (slice)
Output: Top 5: ["quick", "brown", "fox", "jumps", "over"]
```

**Code Structure:** (400+ lines)
- Helper functions: `is_letter`, `char_to_lowercase`, `normalize_word`, `is_stopword`
- Core pipeline: `split_into_words`, `count_words`, `get_top_words`
- Data structures: `WordCount { word: string, count: int }`
- Complete shadow test coverage
- Detailed documentation of each stage
- Real-world applications section

**Learning Value:**
- Most accessible example (everyone understands word counting)
- Clear input/output transformation
- Shows practical use of higher-order functions
- Demonstrates string processing patterns

---

### 2. CSV/TSV Data Processor (`nl_csv_processor.nano`)
**Priority:** HIGH (most requested real-world use case)

**Problem Statement:**  
Parse CSV data, filter rows by criteria, transform values, and compute aggregates. Essential for data analysis, reporting, and ETL pipelines.

**What It Demonstrates:**
- String splitting and parsing (CSV format handling)
- Map for row transformation (apply formulas, convert types)
- Filter for selection (WHERE-like clauses: age > 25, salary > 50000)
- Fold for aggregation (SUM, AVG, COUNT, MIN, MAX)
- Struct operations with real data

**Example Pipeline:**
```
Input CSV:
name,age,salary,department
Alice,30,75000,Engineering
Bob,25,65000,Sales
Carol,35,85000,Engineering
Dave,28,70000,Sales

Pipeline:
  ↓ parse_csv → List<Employee>
  ↓ filter(department == "Engineering")
  ↓ map(apply_raise 10%)
  ↓ fold(sum salaries)

Output:
Filtered: 2 employees
Total salaries: $176,000
Average: $88,000
```

**Data Structures:**
```nano
struct Employee {
    name: string,
    age: int,
    salary: int,
    department: string
}

struct AggregateResult {
    count: int,
    sum: int,
    average: int,
    min: int,
    max: int
}
```

**Real-World Applications:**
- Sales report generation
- Scientific data analysis
- Business intelligence dashboards
- Data migration and ETL

---

### 3. Log File Analyzer (`nl_log_analyzer.nano`)
**Priority:** HIGH (DevOps relevance)

**Problem Statement:**  
Parse application logs, filter by severity level, count error patterns, and identify the most common issues. Critical for debugging and monitoring.

**What It Demonstrates:**
- Pattern matching with string operations
- Map/filter pipeline for log processing
- Fold for counting and grouping
- Practical error analysis techniques

**Example Pipeline:**
```
Input Logs:
[2024-01-01 10:00:00] [ERROR] Failed to connect to database
[2024-01-01 10:00:05] [INFO] Server started on port 8080
[2024-01-01 10:00:10] [ERROR] Timeout waiting for response
[2024-01-01 10:00:15] [WARN] High memory usage detected
[2024-01-01 10:00:20] [ERROR] Failed to connect to database

Pipeline:
  ↓ parse_log_lines → List<LogEntry>
  ↓ filter(level == ERROR)
  ↓ map(extract_error_message)
  ↓ fold(count_by_pattern)

Output:
Total errors: 3
Error patterns:
  - "Failed to connect to database": 2 occurrences
  - "Timeout waiting for response": 1 occurrence
Most common: "Failed to connect to database"
```

**Data Structures:**
```nano
enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4
}

struct LogEntry {
    timestamp: string,
    level: LogLevel,
    message: string
}

struct ErrorPattern {
    pattern: string,
    count: int,
    first_seen: string,
    last_seen: string
}
```

**Real-World Applications:**
- Production monitoring
- Incident response
- Security analysis
- Performance debugging

---

### 4. Sales Data Pipeline (`nl_sales_pipeline.nano`)
**Priority:** MEDIUM (business analytics showcase)

**Problem Statement:**  
Process sales transactions: filter by region, apply discounts, compute totals, and identify top-performing products. Demonstrates business intelligence workflows.

**What It Demonstrates:**
- Chaining map/filter/fold operations
- Working with complex structs
- List<T> with user-defined types
- Multi-stage data transformation
- Business logic implementation

**Example Pipeline:**
```
Input: List<Sale>
Sale { product: "Laptop", amount: 1200, region: "West", date: "2024-01-01" }
Sale { product: "Mouse", amount: 25, region: "East", date: "2024-01-01" }
Sale { product: "Laptop", amount: 1200, region: "West", date: "2024-01-02" }
...

Pipeline:
  ↓ filter(region == "West")
  ↓ map(apply_seasonal_discount 15%)
  ↓ fold(sum by product)
  ↓ sort by total descending
  ↓ take top 10

Output:
West Region Sales (with 15% discount):
  1. Laptop: $2,040 (2 units)
  2. Monitor: $850 (3 units)
  ...
Total revenue: $15,340
```

**Real-World Applications:**
- Sales reporting
- Revenue forecasting
- Product performance analysis
- Regional comparisons

---

### 5. AST Code Analyzer (`nl_ast_analyzer.nano`)
**Priority:** MEDIUM (advanced metaprogramming)

**Problem Statement:**  
Analyze NanoLang source code to compute metrics: function count, call graph, cyclomatic complexity, unused variables. Demonstrates static analysis capabilities.

**What It Demonstrates:**
- AST traversal with recursion
- Pattern matching on AST nodes
- Fold for metrics aggregation
- Practical metaprogramming
- Building developer tools

**Example Analysis:**
```
Input: NanoLang source code (as AST)

Analysis Pipeline:
  ↓ traverse AST recursively
  ↓ filter(node_type == FUNCTION_DEF)
  ↓ map(extract_function_info)
  ↓ fold(compute_metrics)

Output:
Code Metrics:
  - Total functions: 15
  - Average function length: 12 lines
  - Cyclomatic complexity: 3.2 average
  - Unused variables: 2
  - Function calls: 47
  - Most called: println (12 times)

Call Graph:
  main → process_data → validate_input
       → format_output
```

**Real-World Applications:**
- Static analysis tools
- Code quality metrics
- Refactoring tools
- Documentation generation
- Linters and formatters

---

## Pedagogical Principles Applied

### 1. Problem-First Approach
Start with a relatable problem that developers encounter in real work. Show the challenge before the solution.

### 2. Real-World Relevance
Every example maps to actual industry use cases. Include sections on "Real-World Applications" and "When to Use This."

### 3. Progressive Complexity
Order examples from simple (word counting) to complex (AST analysis). Build on concepts from previous examples.

### 4. Clear Input/Output
Show concrete examples of data transformation. Use realistic data, not `[1, 2, 3, 4, 5]`.

### 5. Comprehensive Documentation
Explain **WHY** each step exists, not just **HOW** it works. Include:
- Problem statement
- Pipeline stages with diagrams
- Data structure rationale
- Performance considerations
- Extension suggestions

### 6. Complete Shadow Tests
Every function has shadow tests. Tests serve as additional documentation of expected behavior.

### 7. Performance Notes
Discuss trade-offs (e.g., linear search vs. hash map, in-place vs. functional updates).

---

## Research Sources

This plan is based on web research of:
- **Functional programming textbooks:** SICP-style problem-solving approaches
- **GitHub examples:** Real-world map/reduce/filter applications
- **Language tutorials:** Python, C#, JavaScript pedagogical examples
- **Classic CS problems:** Word frequency, log parsing, data pipelines, CSV processing

Key insight: The best teaching examples solve **one clear problem** that students recognize from their own experience.

---

## Implementation Checklist

### For Each Example:
- [ ] Problem statement (2-3 paragraphs)
- [ ] Real-world applications section
- [ ] Pipeline diagram (text-based)
- [ ] Data structure definitions
- [ ] Helper functions with shadow tests
- [ ] Core pipeline functions with shadow tests
- [ ] Main demonstration with realistic data
- [ ] Performance notes
- [ ] Extension suggestions
- [ ] 300-500 lines total
- [ ] Compiles without warnings
- [ ] All shadow tests pass

---

## Success Metrics

1. **Clarity:** Can a developer unfamiliar with NanoLang understand the problem and solution?
2. **Practicality:** Can they adapt the example to their own use case?
3. **Completeness:** Are all steps explained and tested?
4. **Realism:** Does it use realistic data and scenarios?
5. **Teaching:** Does it explain WHY, not just HOW?

---

## Next Steps

1. ✅ Complete `nl_word_frequency.nano` (90% done, debugging string comparisons)
2. Implement `nl_csv_processor.nano` (highest demand)
3. Create `nl_log_analyzer.nano` (DevOps value)
4. Build `nl_sales_pipeline.nano` (business showcase)
5. Develop `nl_ast_analyzer.nano` (advanced capabilities)

Each example will serve as both:
- **Tutorial:** Teaching how to use the features
- **Template:** Starting point for real projects
- **Showcase:** Demonstrating NanoLang's capabilities

---

## Appendix: Additional Example Ideas

**Medium Priority:**
- JSON-like data transformer (nested structure manipulation)
- Text processing pipeline (NLP preprocessing)
- Student grade analyzer (education domain)
- Network packet filter (systems programming)
- Tree operations (recursive data structures)

**Lower Priority:**
- Configuration file parser
- Markdown to HTML converter
- Simple expression evaluator
- File system analyzer
- Test result aggregator

