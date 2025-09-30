# nanolang Roadmap

This document outlines the development roadmap for nanolang.

## Project Vision

Build a minimal, LLM-friendly programming language that:
- Compiles to C for performance and portability
- Requires shadow-tests for all code
- Uses unambiguous prefix notation
- Eventually self-hosts (compiles itself)

## Current Status: Phase 0 - Specification ✅

**Status**: Complete

**Deliverables**:
- ✅ Language specification document
- ✅ Grammar definition
- ✅ Type system design
- ✅ Shadow-test semantics
- ✅ Example programs
- ✅ Getting started guide
- ✅ Documentation

## Phase 1 - Lexer (Not Started)

**Goal**: Transform source text into tokens

**Deliverables**:
- [ ] Token definitions
- [ ] Lexer implementation
- [ ] Error reporting
- [ ] Test suite for lexer
- [ ] Handle comments
- [ ] Handle string literals
- [ ] Handle numeric literals

**Estimated Effort**: 2-3 weeks

**Success Criteria**:
- Can tokenize all example programs
- Clear error messages for invalid input
- 100% test coverage

## Phase 2 - Parser (Not Started)

**Goal**: Transform tokens into Abstract Syntax Tree (AST)

**Deliverables**:
- [ ] AST node definitions
- [ ] Recursive descent parser
- [ ] Operator precedence (prefix notation)
- [ ] Error recovery
- [ ] Test suite for parser
- [ ] Pretty-printer (AST → source)

**Estimated Effort**: 3-4 weeks

**Success Criteria**:
- Can parse all example programs
- Produces valid AST
- Helpful error messages
- Round-trip: source → AST → source

## Phase 3 - Type Checker (Not Started)

**Goal**: Verify type correctness of AST

**Deliverables**:
- [ ] Type inference engine
- [ ] Type checking rules
- [ ] Symbol table
- [ ] Scope resolution
- [ ] Error messages for type errors
- [ ] Test suite for type checker

**Estimated Effort**: 3-4 weeks

**Success Criteria**:
- Catches all type errors
- Rejects invalid programs
- Accepts valid programs
- Clear error messages

## Phase 4 - Shadow-Test Runner (Not Started)

**Goal**: Execute shadow-tests during compilation

**Deliverables**:
- [ ] Test extraction from AST
- [ ] Interpreter for shadow-tests
- [ ] Assertion checking
- [ ] Test result reporting
- [ ] Test coverage tracking
- [ ] Test suite for test runner

**Estimated Effort**: 2-3 weeks

**Success Criteria**:
- Executes all shadow-tests
- Reports failures clearly
- Tracks test coverage
- Fast execution

## Phase 5 - C Transpiler (Not Started)

**Goal**: Transform AST to C code

**Deliverables**:
- [ ] C code generation
- [ ] Runtime library
- [ ] Built-in function implementations
- [ ] Memory management
- [ ] Test suite for transpiler
- [ ] C code formatter

**Estimated Effort**: 4-5 weeks

**Success Criteria**:
- Generates valid C code
- Compiles with standard C compiler
- Matches nanolang semantics
- Readable output

## Phase 6 - Standard Library (Not Started)

**Goal**: Provide common functionality

**Deliverables**:
- [ ] String operations
- [ ] I/O functions
- [ ] Math functions
- [ ] Data structures (arrays, lists)
- [ ] Documentation
- [ ] Shadow-tests for all functions

**Estimated Effort**: 3-4 weeks

**Success Criteria**:
- Well-documented
- Fully tested
- Useful for real programs
- Consistent API

## Phase 7 - Command-Line Tool (Not Started)

**Goal**: User-friendly compiler interface

**Deliverables**:
- [ ] `nanoc` compiler command
- [ ] Command-line options
- [ ] Help system
- [ ] Error formatting
- [ ] Build configuration
- [ ] Documentation

**Estimated Effort**: 2 weeks

**Success Criteria**:
- Easy to use
- Clear error messages
- Good help text
- Follows Unix conventions

## Phase 8 - Self-Hosting (Not Started)

**Goal**: Compile nanolang compiler in nanolang

**Deliverables**:
- [ ] Rewrite compiler in nanolang
- [ ] Bootstrap process
- [ ] Performance optimization
- [ ] Documentation
- [ ] Test suite

**Estimated Effort**: 8-12 weeks

**Success Criteria**:
- nanolang compiles itself
- Bootstrapping works
- Performance acceptable
- All tests pass

## Future Enhancements

These features may be added after self-hosting:

### Language Features
- [ ] Arrays and slices
- [ ] Structs/records
- [ ] Generics/templates
- [ ] Pattern matching
- [ ] Modules/imports
- [ ] Error handling (Result type)
- [ ] Algebraic data types

### Tooling
- [ ] REPL (Read-Eval-Print Loop)
- [ ] Language server (LSP)
- [ ] Debugger
- [ ] Package manager
- [ ] Build system
- [ ] Documentation generator

### Optimizations
- [ ] Tail call optimization
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] Inlining
- [ ] LLVM backend (alternative to C)

### Ecosystem
- [ ] VS Code extension
- [ ] Vim plugin
- [ ] Emacs mode
- [ ] Online playground
- [ ] Tutorial website
- [ ] Community forum

## Timeline Estimate

| Phase | Estimated Duration | Dependencies |
|-------|-------------------|--------------|
| Phase 0: Specification | Complete | - |
| Phase 1: Lexer | 2-3 weeks | Phase 0 |
| Phase 2: Parser | 3-4 weeks | Phase 1 |
| Phase 3: Type Checker | 3-4 weeks | Phase 2 |
| Phase 4: Shadow-Test Runner | 2-3 weeks | Phase 3 |
| Phase 5: C Transpiler | 4-5 weeks | Phase 4 |
| Phase 6: Standard Library | 3-4 weeks | Phase 5 |
| Phase 7: CLI Tool | 2 weeks | Phase 5 |
| Phase 8: Self-Hosting | 8-12 weeks | Phase 7 |

**Total Estimated Time**: 6-9 months for basic self-hosting implementation

## Milestones

### Milestone 1: First Compilation (Phase 1-5)
**Target**: Basic compiler working
- Can compile simple nanolang programs
- Generates working C code
- Shadow-tests execute

### Milestone 2: Usable Compiler (Phase 6-7)
**Target**: Ready for real projects
- Standard library available
- Command-line tool polished
- Documentation complete

### Milestone 3: Self-Hosting (Phase 8)
**Target**: nanolang compiles itself
- Compiler rewritten in nanolang
- Bootstrap process working
- Full test suite passing

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

**Current Focus**: Implementation planning

**Most Needed**:
1. Feedback on specification
2. Additional example programs
3. Test cases
4. Implementation volunteers

## Success Metrics

### Technical
- All example programs compile and run
- Shadow-tests catch bugs
- Generated C code is readable
- Compilation is fast
- Self-hosting works

### Community
- Clear documentation
- Active contributors
- Growing example library
- Positive feedback

### Adoption
- Real projects using nanolang
- LLMs can generate correct code
- Teaching material available
- Community resources

## Risks and Mitigations

### Risk: Specification Changes
**Mitigation**: Community review before implementation starts

### Risk: Implementation Complexity
**Mitigation**: Incremental development, extensive testing

### Risk: Performance Issues
**Mitigation**: C transpilation provides good baseline performance

### Risk: Limited Contributors
**Mitigation**: Keep codebase simple and well-documented

### Risk: LLM Generation Quality
**Mitigation**: Iterate on language design based on LLM testing

## Communication

### Updates
- Commit messages
- Release notes
- GitHub issues/PRs

### Discussion
- GitHub Discussions (when available)
- Issue tracker for bugs/features

### Documentation
- Keep docs in sync with code
- Update examples regularly
- Maintain changelog

## Versioning

Following semantic versioning (semver):

- **0.x.y**: Pre-1.0 development
- **1.0.0**: First stable release (after self-hosting)
- **1.x.0**: New features (backwards compatible)
- **x.0.0**: Breaking changes

## Release Strategy

### Pre-1.0 Releases
- 0.1.0: Lexer complete
- 0.2.0: Parser complete
- 0.3.0: Type checker complete
- 0.4.0: Shadow-test runner complete
- 0.5.0: C transpiler complete
- 0.6.0: Standard library complete
- 0.7.0: CLI tool complete
- 0.9.0: Self-hosting beta

### 1.0 Release Criteria
- Self-hosting works
- All examples compile
- Documentation complete
- Test suite passes
- Performance acceptable
- Breaking changes unlikely

## Long-Term Vision

nanolang aims to be:

1. **Reference implementation** for LLM-friendly language design
2. **Teaching tool** for programming language concepts
3. **Practical language** for systems programming
4. **Proof of concept** for shadow-test methodology
5. **Community project** with active contributors

## Questions?

For questions about the roadmap:
1. Check [SPECIFICATION.md](SPECIFICATION.md) for language details
2. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help
3. Open an issue for discussion

---

**Last Updated**: Initial roadmap  
**Next Review**: After Phase 1 completion
