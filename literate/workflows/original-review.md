---
schema: "literate-ai/generation-workflow-markdown@1"
workflow_id: "nanolang-original-review"
version: "1.0.0"
stages:
  - stage_id: "plan"
    kind: "model"
    dependencies: []
    response_schema_name: "sample_implementation_plan"
    content_kind: "metadata"
    required_capabilities:
      - "structured-output"
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "generate"
    kind: "model"
    dependencies:
      - "plan"
    response_schema_name: "sample_source_tree"
    content_kind: "source"
    required_capabilities:
      - "structured-output"
      - "source-generation"
    produces_tree: true
    maximum_output_tokens: null
  - stage_id: "validate"
    kind: "lifecycle"
    dependencies:
      - "generate"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "classify"
    kind: "lifecycle"
    dependencies:
      - "validate"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "authorize-build"
    kind: "lifecycle"
    dependencies:
      - "classify"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "build"
    kind: "lifecycle"
    dependencies:
      - "authorize-build"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "resolve-dependencies"
    kind: "lifecycle"
    dependencies:
      - "build"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "test-generated"
    kind: "lifecycle"
    dependencies:
      - "resolve-dependencies"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "verify-independent"
    kind: "lifecycle"
    dependencies:
      - "test-generated"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "prepare-tree"
    kind: "lifecycle"
    dependencies:
      - "verify-independent"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
  - stage_id: "commit-tree"
    kind: "lifecycle"
    dependencies:
      - "prepare-tree"
    response_schema_name: ""
    content_kind: "metadata"
    required_capabilities: []
    produces_tree: false
    maximum_output_tokens: null
---
# My original-source review workflow

I retain the framework stage protocol for future reviewed planning. I do not execute this candidate-generating
workflow on my original source. Original adoption does not admit a generated replacement. No provider group,
lifecycle driver or host-execution authorization is configured. My native unit/profile and test policy milestones
must close before retained execution; regenerative work needs separate authority review.

## Stage: plan

I plan only against the exact reviewed native source, ownership and acceptance contract. I identify unresolved
producer/toolchain/export boundaries explicitly. I do not infer a new implementation language or execute a plan
while original-source authority remains unqualified for generation.

## Stage: generate

I require separately reviewed regenerative authority before generating any replacement. I preserve the exact
approved behavior and refusal contract and produce an isolated candidate only after such authority exists.
