# NanoLang User Guide

**Welcome to NanoLang by Example** - A comprehensive guide to writing NanoLang programs.

## What is NanoLang?

NanoLang is a compiled systems programming language designed specifically for LLM code generation. It transpiles to C for native performance while maintaining a simple, consistent syntax.

**Key Features:**
- **LLM-First Design:** Exactly ONE canonical way to write each construct
- **Prefix Notation:** All operations use `(f x y)` form
- **Explicit Types:** Always annotate types, minimal inference
- **Shadow Tests:** Every function has compile-time tests (mandatory)
- **C Interop:** Full FFI for calling C libraries
- **Zero Runtime Overhead:** Compiles to native C code

## How to Use This Guide

This guide is organized into four main parts:

### Part I: Language Fundamentals
Learn the core NanoLang language, one concept at a time. Start here if you're new to NanoLang.

- **Chapter 1:** Getting Started
- **Chapter 2:** Basic Syntax & Types
- **Chapter 3:** Variables & Bindings
- **Chapter 4:** Functions
- **Chapter 5:** Control Flow
- **Chapter 6:** Collections
- **Chapter 7:** Data Structures
- **Chapter 8:** Modules & Imports

### Part II: Standard Library
The batteries included with NanoLang - built-in utilities available in every program.

- **Chapter 9:** Core Utilities
- **Chapter 10:** Collections Library
- **Chapter 11:** I/O & Filesystem
- **Chapter 12:** System & Runtime

### Part III: External Modules
Powerful extensions for graphics, networking, databases, and more.

- **Chapter 13:** Text Processing (regex, log, StringBuilder)
- **Chapter 14:** Data Formats (JSON, SQLite)
- **Chapter 15:** Web & Networking (curl, http_server, uv)
- **Chapter 16:** Graphics Fundamentals (SDL modules)
- **Chapter 17:** OpenGL Graphics
- **Chapter 18:** Game Development
- **Chapter 19:** Terminal UI (ncurses)
- **Chapter 20:** Testing & Quality
- **Chapter 21:** Configuration

### Part IV: Advanced Topics
Deep dives into best practices, patterns, and performance.

- **Chapter 22:** Canonical Style Guide
- **Chapter 23:** Higher-Level Patterns
- **Chapter 24:** Performance & Optimization
- **Chapter 25:** Contributing & Extending

### Appendices
Quick references, troubleshooting, and comprehensive examples.

- **Appendix A:** Examples Gallery
- **Appendix B:** Quick Reference
- **Appendix C:** Troubleshooting Guide
- **Appendix D:** Glossary

## Learning Path

**Absolute Beginners:** Start with Part I and work through sequentially.

**Experienced Programmers:** Skim Chapters 1-5, focus on NanoLang-specific features (shadow tests, prefix notation, modules).

**Looking for Specific Features:** Use the Quick Reference (Appendix B) or search functionality.

**Working on a Project:** Part III shows you how to use external libraries for graphics, networking, databases, and more.

## Example-Driven Documentation

Every function, every feature has working examples with shadow tests. You can copy and run any code snippet in this guide.

**Conventions:**
- ✅ **Try This:** Working example you can run
- ⚠️ **Watch Out:** Common pitfalls and warnings
- 💡 **Pro Tip:** Best practices and optimization hints
- ❌ **Don't Do This:** Anti-patterns to avoid

## Get Started

Ready to write your first NanoLang program? Jump to [Chapter 1: Getting Started](part1_fundamentals/01_getting_started.md).

Already familiar with the basics? Explore the [Examples Gallery](appendices/a_examples_gallery.md) for real-world programs.

---

**NanoLang Version:** 0.5+ (current development version)  
**Last Updated:** 2026-01-20
