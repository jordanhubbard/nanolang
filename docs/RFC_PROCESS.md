# How I Evolve

**RFC = Request for Comments**

## My Evolution

My RFC process is how I evolve. It is the method I use to make decisions about my syntax, my features, and any changes that might break existing code. I use this process to remain transparent and to ensure that any growth follows my core values.

## When I Require an RFC

### I Require an RFC for:

- **Changes to My Language**
  - New keywords or syntax
  - Changes to my type system
  - New control flow constructs
  - Breaking changes to features I already have

- **Major Features**
  - New data structures, such as Set<T> or Map<K,V>
  - Additions to my standard library of more than five functions
  - New compilation modes or targets
  - Changes to my FFI system

- **Tooling and Infrastructure**
  - New compiler flags or modes
  - Build system changes that affect you
  - My package manager design
  - IDE integration protocols

- **Process and Policy**
  - Changes to this RFC process
  - Updates to my code of conduct
  - Changes to how people contribute to me

### I Do Not Require an RFC for:

You can handle these through normal pull requests:

- Bug fixes that do not change my behavior
- Improvements to my documentation
- Refactoring my code without changing how you interact with me
- Adding tests
- Performance optimizations that do not change my API
- Small additions to my standard library of one or two functions
- Examples or tutorials

## My RFC Lifecycle

```
[Draft] → [Proposed] → [Discussion] → [Final Comment] → [Decision]
                                           ↓
                              [Accepted] or [Rejected] or [Postponed]
                                           ↓
                              [Implemented] → [Stable]
```

### My Stages

1. **Draft**: The author is writing the RFC and it is not yet ready for me to see.
2. **Proposed**: The RFC is submitted as a pull request to my `rfcs/` directory.
3. **Discussion**: My community discusses the proposal and the author makes revisions.
4. **Final Comment Period (FCP)**: A ten day period for final review.
5. **Decision**: My core team decides whether to accept, reject, or postpone.
6. **Accepted**: The RFC is merged and you can begin implementing it.
7. **Implemented**: The feature is built and merged into my code.
8. **Stable**: The feature is released in a stable version of me.

## How to Propose a Change to Me

### Step 1: Socialize the Idea

Before you write a formal RFC:

1. Open a GitHub Discussion or Issue to describe your idea.
2. See if others are interested.
3. Get early feedback on whether your idea is feasible.
4. Consider other ways to solve the same problem.

**The question to answer:** Is this worth a full RFC?

### Step 2: Write the RFC

1. Fork my repository.
2. Copy `docs/rfcs/0000-template.md` to `docs/rfcs/0000-my-feature.md`.
3. Fill out the template.
4. Write clearly and thoroughly.
5. Include examples and your reasoning.

### Step 3: Submit a Pull Request

1. Submit a pull request with your RFC document.
2. Title it `RFC: Brief Description`.
3. Label it `T-rfc`.
4. Mention relevant maintainers in your description.

### Step 4: Discussion Period

- My community reviews and comments on your proposal.
- You revise the proposal based on what they say.
- Discussion can take days or weeks.
- If you make major changes, I might require a new discussion period.

### Step 5: Final Comment Period (FCP)

When the discussion settles:

1. A member of my core team moves the RFC to FCP.
2. The label changes to `T-rfc-fcp`.
3. A ten day countdown begins.
4. This is the last chance to raise concerns.

### Step 6: Decision

After the FCP, my core team makes a choice:

- **Accept**: The RFC is merged and implementation begins.
- **Reject**: The RFC is closed and I explain why.
- **Postpone**: The idea is good but the timing is wrong.

## My RFC Template

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

## How I Make Decisions

My core team considers these factors:

### Technical Merit
- Does it solve a real problem?
- Is the design sound?
- Can it be implemented?
- What is the impact on my performance?
- How does it interact with my existing features?

### Alignment with My Goals
- Is it consistent with my philosophy?
- Does it maintain my simplicity?
- Is it friendly to machines and LLMs?
- Does it fit my mental model?

### Community Support
- Do my users want this?
- Is there a consensus?
- Are my maintainers willing to support it?

### Cost and Benefit
- How much effort is needed to implement it?
- What is the maintenance burden?
- How much documentation is required?
- Are breaking changes justified?

## My Core Team

These are the people with authority to make RFC decisions for me:

- Jordan Hubbard (@jordanhubbard) - My creator and lead

As I mature, I will expand my core team.

## How I Number RFCs

I number RFCs sequentially:

- `0001-feature-name.md`
- `0002-another-feature.md`

I assign these numbers when I accept an RFC, not when it is proposed.

## Examples of Changes I Might Consider

### Language Features
- `RFC: Add 'defer' statement for cleanup`
- `RFC: Introduce 'range' type for iteration`
- `RFC: Add pattern matching on integers`
- `RFC: Support method syntax (dot notation)`

### My Standard Library
- `RFC: Add Result<T,E> helper functions`
- `RFC: Standard JSON parsing library`
- `RFC: Network socket API`
- `RFC: Regular expression support`

### My Tooling
- `RFC: Package manager design`
- `RFC: Language server protocol (LSP)`
- `RFC: Built-in testing framework`
- `RFC: Code formatter specification`

## Questions You Might Have

### Q: How long does my RFC process take?

It varies:
- Simple features: two to four weeks.
- Complex features: one to three months.
- Controversial features: three to six months.

### Q: Can I implement a feature before I accept the RFC?

You can, but:
- Your work might be wasted if I reject the RFC.
- A proof of concept can help the discussion.
- You should mark your pull request as "[WIP] RFC Implementation".

### Q: What if I reject your RFC?

- It does not mean your idea was bad.
- It might be the right idea at the wrong time.
- You can change it and submit it again.
- We can look at it again later.

### Q: Can I withdraw my RFC?

You can withdraw your proposal at any time by adding a comment and closing the pull request.

### Q: Who makes the final decision?

My core team decides by consensus. If they cannot agree, my lead maintainer makes the final choice.

### Q: Can an RFC be changed after I accept it?

Yes. You can submit a new RFC or a pull request to amend it. If the changes are significant, I require a new RFC.

## How I Organize My RFCs

```
docs/
└── rfcs/
    ├── 0000-template.md          # My template for new RFCs
    ├── README.md                  # This document
    ├── accepted/
    │   ├── 0001-feature.md
    │   └── 0002-another.md
    ├── rejected/
    │   ├── 0042-bad-idea.md
    └── postponed/
        └── 0123-future-idea.md
```

## What Inspired Me

My process was inspired by:
- The Rust RFC Process
- Python PEPs
- Swift Evolution

## Changing This Process

You can change how I evolve by submitting an RFC for this process itself.

---

**Last Updated:** February 20, 2026  
**Status:** Active  
**Version:** 1.0

