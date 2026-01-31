# NanoLang RFC Process

**RFC = Request for Comments**

## Overview

The NanoLang RFC (Request for Comments) process is how we make decisions about language evolution, major features, and breaking changes. It provides a transparent, collaborative way for the community to propose and discuss changes.

## When to Use the RFC Process

### Requires RFC

Use the RFC process for:

- **Language Changes**
  - New keywords or syntax
  - Changes to type system
  - New control flow constructs
  - Breaking changes to existing features

- **Major Features**
  - New data structures (e.g., Set<T>, Map<K,V>)
  - Standard library additions (>5 functions)
  - New compilation modes or targets
  - FFI system changes

- **Tooling & Infrastructure**
  - New compiler flags or modes
  - Build system changes affecting users
  - Package manager design
  - IDE integration protocols

- **Process & Policy**
  - Changes to the RFC process itself
  - Code of conduct updates
  - Contribution guidelines changes

### Does NOT Require RFC

These can be done via normal pull requests:

- Bug fixes (no behavior change)
- Documentation improvements
- Code refactoring (no user-facing changes)
- Test additions
- Performance optimizations (no API change)
- Small stdlib additions (1-2 functions)
- Examples or tutorials

## RFC Lifecycle

```
[Draft] → [Proposed] → [Discussion] → [Final Comment] → [Decision]
                                           ↓
                              [Accepted] or [Rejected] or [Postponed]
                                           ↓
                              [Implemented] → [Stable]
```

### Stages

1. **Draft** - Author is writing the RFC, not yet ready for review
2. **Proposed** - RFC submitted as PR to rfcs/ directory
3. **Discussion** - Community discusses, author revises
4. **Final Comment Period (FCP)** - 10-day final review period
5. **Decision** - Core team makes decision (Accept/Reject/Postpone)
6. **Accepted** - RFC merged, implementation can begin
7. **Implemented** - Feature implemented and merged
8. **Stable** - Feature released in stable version

## How to Submit an RFC

### Step 1: Socialize the Idea

Before writing a formal RFC:

1. Open a GitHub Discussion or Issue describing the idea
2. Gauge community interest
3. Get early feedback on feasibility
4. Consider alternatives

**Question to answer:** "Is this worth a full RFC?"

### Step 2: Write the RFC

1. Fork the repository
2. Copy `docs/rfcs/0000-template.md` to `docs/rfcs/0000-my-feature.md`
3. Fill out the template (see below)
4. Write clearly and thoroughly
5. Include examples and rationale

### Step 3: Submit Pull Request

1. Submit PR with RFC document
2. Title: `RFC: Brief Description`
3. Label: `T-rfc`
4. Ping relevant maintainers in PR description

### Step 4: Discussion Period

- Community reviews and comments
- Author revises based on feedback
- Discussion can take days to weeks
- Major changes may require new discussion period

### Step 5: Final Comment Period (FCP)

When discussion settles:

1. Core team member moves RFC to FCP
2. Label changes to `T-rfc-fcp`
3. 10-day countdown begins
4. Last chance for concerns

### Step 6: Decision

After FCP, core team decides:

- **Accept** - RFC merged, implementation begins
- **Reject** - RFC closed with explanation
- **Postpone** - Good idea but not right time

## RFC Template

```markdown
# RFC: [Feature Name]

- RFC PR: #XXXX
- Status: [Draft/Proposed/FCP/Accepted/Rejected/Implemented]
- Author: [Your Name] (@github-username)
- Created: YYYY-MM-DD

## Summary

One paragraph explanation of the feature.

## Motivation

Why are we doing this? What use cases does it support? What problems does it solve?

## Guide-Level Explanation

Explain the proposal as if teaching it to a NanoLang user. Include:
- How users will use this feature
- Code examples
- Common use cases
- How it interacts with existing features

## Reference-Level Explanation

Technical details:
- Precise syntax and semantics
- How it's implemented (high-level)
- Edge cases
- Error handling
- Interaction with other features

## Drawbacks

Why should we NOT do this?
- Complexity added
- Maintenance burden
- Learning curve
- Compatibility issues

## Alternatives

What other approaches were considered?
- Why were they not chosen?
- Trade-offs compared to this proposal

## Prior Art

What do other languages do?
- Rust
- Go
- Python
- C
- Others

## Unresolved Questions

What parts need more design work?
- Open questions
- Future extensions
- Out of scope

## Future Possibilities

What might we add later?
- Natural extensions
- Related features
- Long-term vision
```

## Decision Criteria

Core team considers:

### Technical Merit
- Does it solve a real problem?
- Is the design sound?
- Is it implementable?
- Performance impact?
- Interaction with existing features?

### Alignment with Goals
- Consistent with NanoLang philosophy?
- Maintains simplicity?
- LLM-friendly?
- Fits mental model?

### Community Support
- Do users want this?
- Is there consensus?
- Are maintainers willing to support it?

### Cost vs Benefit
- Implementation effort
- Maintenance burden
- Documentation needed
- Breaking changes justified?

## Core Team

Current core team members with RFC decision authority:

- Jordan Hubbard (@jordanhubbard) - Creator/Lead

**Note:** As project matures, core team will expand.

## RFC Numbering

RFCs are numbered sequentially:

- `0001-feature-name.md`
- `0002-another-feature.md`
- etc.

Numbers assigned when RFC is accepted (not when proposed).

## Examples of Potential RFCs

### Language Features
- `RFC: Add 'defer' statement for cleanup`
- `RFC: Introduce 'range' type for iteration`
- `RFC: Add pattern matching on integers`
- `RFC: Support method syntax (dot notation)`

### Standard Library
- `RFC: Add Result<T,E> helper functions`
- `RFC: Standard JSON parsing library`
- `RFC: Network socket API`
- `RFC: Regular expression support`

### Tooling
- `RFC: Package manager design`
- `RFC: Language server protocol (LSP)`
- `RFC: Built-in testing framework`
- `RFC: Code formatter specification`

## FAQ

### Q: How long does the RFC process take?

Varies widely:
- Simple features: 2-4 weeks
- Complex features: 1-3 months
- Controversial features: 3-6 months

### Q: Can I implement before RFC is accepted?

Yes, but:
- Implementation may be wasted if RFC rejected
- Proof-of-concept implementations can help discussion
- Mark PR as "[WIP] RFC Implementation"

### Q: What if my RFC is rejected?

- Not a reflection on you!
- May be right idea, wrong time
- Can resubmit with changes
- Can revisit later

### Q: Can I withdraw my RFC?

Yes, at any time. Add comment and close PR.

### Q: Who makes the final decision?

Core team makes decision by consensus. If no consensus, lead maintainer has final say.

### Q: Can RFCs be amended after acceptance?

Yes, via new RFC or amendment PR. Significant changes require new RFC.

## RFC Repository Structure

```
docs/
└── rfcs/
    ├── 0000-template.md          # Template for new RFCs
    ├── README.md                  # This document
    ├── accepted/
    │   ├── 0001-feature.md
    │   └── 0002-another.md
    ├── rejected/
    │   └── 0042-bad-idea.md
    └── postponed/
        └── 0123-future-idea.md
```

## References

This process inspired by:
- [Rust RFC Process](https://github.com/rust-lang/rfcs)
- [Python PEPs](https://www.python.org/dev/peps/)
- [Swift Evolution](https://github.com/apple/swift-evolution)

## Changes to This Process

This process itself can be changed via RFC!

---

**Last Updated:** January 25, 2026  
**Status:** Active  
**Version:** 1.0
