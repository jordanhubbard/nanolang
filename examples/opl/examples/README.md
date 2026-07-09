# OPL Examples

This directory contains example `.opl` files demonstrating various features of the Operator Prompt Language (OPL).

## Available Examples

### 1. `hello.opl` - Hello World Task
A simple task that greets a user with string concatenation and emit.

**Features demonstrated:**
- Task definition with `task` keyword
- Input parameters with default values
- String concatenation with `+`
- Variable binding with `let`
- Event emission with `emit`

### 2. `web_search.opl` - Basic Agent with Tool Call
An agent that searches the web using a tool.

**Features demonstrated:**
- Agent definition with `agent` keyword
- Tool declaration with `uses`
- Tool calling with `call...as`
- Assertions with `assert...else`
- Result emission

### 3. `conditional.opl` - Conditional Logic
An agent with conditional execution using `when`.

**Features demonstrated:**
- Conditional execution with `when`
- Guard expressions (`!=`)
- Inline statement sequences

### 4. `multi_tool.opl` - Multiple Tool Calls
An agent that chains multiple tool calls together.

**Features demonstrated:**
- Multiple tool declarations
- Sequential tool calls
- Passing results between tools
- Result validation with assertions

### 5. `service_spec.opl` - Service Specifications
Service definitions (non-executable specifications).

**Features demonstrated:**
- Service definitions with `service` keyword
- Input/output specifications
- Documentation with `doc`
- Multiple services in one file

## Building and Running Examples

### Prerequisites

Compile the OPL CLI:

```bash
make cli
```

### Parse Examples

Parse an OPL file to AST JSON:

```bash
make parse EXAMPLE=hello
```

### Validate Examples

Check an OPL file for semantic errors:

```bash
make validate EXAMPLE=hello
```

### Compile Examples

Compile an OPL file to PLAN_IR:

```bash
make compile EXAMPLE=hello
```

### Generate Code

Generate NanoLang skeleton from PLAN_IR:

```bash
make codegen EXAMPLE=hello
```

### Build All Examples

Parse, validate, and compile all examples:

```bash
make all
```

### Clean Generated Files

Remove all generated files:

```bash
make clean
```

## Output Files

Generated files are placed in the `output/` subdirectory:
- `output/<name>.ast.json` - Abstract Syntax Tree
- `output/<name>.validate.json` - Validation results
- `output/<name>.plan.json` - PLAN_IR (compiled output)
- `output/<name>.nano` - Generated NanoLang skeleton

## OPL Language Reference

### Agent Definition

```opl
agent <name> {
  doc "<description>"
  uses <tool1>, <tool2>
  input <name>:<type>, ...
  
  # Statements
}
```

### Task Definition

```opl
task <name> {
  doc "<description>"
  input <name>:<type> = <default>
  
  # Statements
}
```

### Service Definition

```opl
service <name> {
  doc "<description>"
  input <name>:<type>
  output <name>:<type>
}
```

### Statements

- **Tool call:** `call <tool> { <args> } as <var>`
- **Variable binding:** `let <var> = <expr>`
- **Assertion:** `assert <expr> else "<message>"`
- **Emit:** `emit <name>: <expr>`
- **Conditional:** `when <expr> -> <stmt>, <stmt>`

### Types

- `string` - Text strings
- `int` - Integers
- `bool` - Booleans (true/false)

## More Information

See the main OPL README at `../README.md` for full toolchain documentation.
