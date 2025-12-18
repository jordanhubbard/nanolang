# Nanolang Web Playground Design

## Overview

An interactive web-based environment for writing, compiling, and running nanolang code directly in the browser. No installation required.

## Goals

1. **Learning Tool**: Help newcomers explore nanolang interactively
2. **Share Code**: Easy sharing of code snippets via URLs
3. **Accessibility**: Works on any device with a web browser
4. **Full-Featured**: Support for compilation, execution, and debugging
5. **Performance**: Fast compilation and execution

## Architecture

```
┌──────────────────────────────────────┐
│         Frontend (React)             │
│                                      │
│  ┌──────────────┐   ┌─────────────┐ │
│  │    Monaco    │   │   Console   │ │
│  │    Editor    │   │   Output    │ │
│  └──────┬───────┘   └──────┬──────┘ │
│         │                  │        │
│         ▼                  ▼        │
│  ┌──────────────────────────────┐  │
│  │      Playground State        │  │
│  └──────────┬───────────────────┘  │
└─────────────┼──────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      WASM Runtime / Backend         │
│                                     │
│  ┌────────────┐    ┌─────────────┐ │
│  │  Compiler  │    │ Interpreter │ │
│  │   (WASM)   │    │   (WASM)    │ │
│  └────────────┘    └─────────────┘ │
└─────────────────────────────────────┘
```

## Frontend Components

### 1. Code Editor

Use Monaco Editor (VS Code's editor):

```typescript
// src/components/Editor.tsx

import * as monaco from 'monaco-editor';

export const Editor: React.FC = () => {
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor>();
  
  useEffect(() => {
    // Register nanolang language
    monaco.languages.register({ id: 'nanolang' });
    
    // Define syntax highlighting
    monaco.languages.setMonarchTokensProvider('nanolang', {
      keywords: ['fn', 'let', 'if', 'while', 'return', ...],
      operators: ['+', '-', '*', '/', '==', '!=', ...],
      // ... more tokens
    });
    
    // Create editor
    editorRef.current = monaco.editor.create(editorDiv, {
      value: defaultCode,
      language: 'nanolang',
      theme: 'vs-dark',
      minimap: { enabled: false },
      automaticLayout: true
    });
  }, []);
  
  return <div ref={editorDiv} className="editor" />;
};
```

### 2. Output Console

```typescript
// src/components/Console.tsx

export const Console: React.FC<{ output: string[] }> = ({ output }) => {
  return (
    <div className="console">
      <div className="console-header">
        <span>Output</span>
        <button onClick={clearOutput}>Clear</button>
      </div>
      <div className="console-output">
        {output.map((line, i) => (
          <div key={i} className="console-line">
            {line}
          </div>
        ))}
      </div>
    </div>
  );
};
```

### 3. Toolbar

```typescript
// src/components/Toolbar.tsx

export const Toolbar: React.FC = () => {
  const { compile, run, share, format } = usePlayground();
  
  return (
    <div className="toolbar">
      <button onClick={compile}>
        <PlayIcon /> Compile
      </button>
      <button onClick={run}>
        <RunIcon /> Run
      </button>
      <button onClick={format}>
        <FormatIcon /> Format
      </button>
      <button onClick={share}>
        <ShareIcon /> Share
      </button>
      
      <select onChange={selectExample}>
        <option>Examples</option>
        <option value="factorial">Factorial</option>
        <option value="fibonacci">Fibonacci</option>
        <option value="hello">Hello World</option>
      </select>
    </div>
  );
};
```

### 4. Settings Panel

```typescript
// src/components/Settings.tsx

export const Settings: React.FC = () => {
  const { settings, updateSettings } = usePlayground();
  
  return (
    <div className="settings">
      <h3>Settings</h3>
      
      <label>
        <input
          type="checkbox"
          checked={settings.autoRun}
          onChange={(e) => updateSettings({ autoRun: e.target.checked })}
        />
        Auto-run on change
      </label>
      
      <label>
        Editor Theme:
        <select value={settings.theme} onChange={changeTheme}>
          <option value="vs-dark">Dark</option>
          <option value="vs-light">Light</option>
        </select>
      </label>
      
      <label>
        Font Size:
        <input
          type="range"
          min="10"
          max="24"
          value={settings.fontSize}
          onChange={(e) => updateSettings({ fontSize: e.target.value })}
        />
      </label>
    </div>
  );
};
```

## WASM Integration

### Compiling Nanolang to WASM

```bash
# Using Emscripten
emcc src/*.c \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_compile_nano", "_run_nano", "_free_result"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' \
  -s MODULARIZE=1 \
  -s EXPORT_NAME='NanolangWASM' \
  -o public/nanolang.js
```

### WASM Module Interface

```c
// src/wasm_interface.c

#include <emscripten.h>

typedef struct {
    char* output;
    char* errors;
    int exit_code;
} ExecutionResult;

EMSCRIPTEN_KEEPALIVE
ExecutionResult* compile_nano(const char* source_code) {
    ExecutionResult* result = malloc(sizeof(ExecutionResult));
    
    // Parse
    Program* prog = parse(source_code);
    if (!prog) {
        result->errors = strdup(get_parse_errors());
        result->exit_code = 1;
        return result;
    }
    
    // Type check
    if (!typecheck(prog)) {
        result->errors = strdup(get_type_errors());
        result->exit_code = 1;
        return result;
    }
    
    // Transpile to C
    char* c_code = transpile(prog);
    result->output = c_code;
    result->exit_code = 0;
    
    return result;
}

EMSCRIPTEN_KEEPALIVE
ExecutionResult* run_nano(const char* source_code) {
    ExecutionResult* result = malloc(sizeof(ExecutionResult));
    
    // Parse and execute with interpreter
    Program* prog = parse(source_code);
    if (!prog) {
        result->errors = strdup(get_parse_errors());
        result->exit_code = 1;
        return result;
    }
    
    // Redirect stdout to buffer
    char* output_buffer = NULL;
    size_t output_size = 0;
    FILE* output_stream = open_memstream(&output_buffer, &output_size);
    stdout = output_stream;
    
    // Execute
    Value return_value = interpret(prog);
    
    // Capture output
    fclose(output_stream);
    result->output = output_buffer;
    result->exit_code = 0;
    
    return result;
}

EMSCRIPTEN_KEEPALIVE
void free_result(ExecutionResult* result) {
    if (result->output) free(result->output);
    if (result->errors) free(result->errors);
    free(result);
}
```

### TypeScript Wrapper

```typescript
// src/wasm/nanolang.ts

interface NanolangModule {
  compile_nano(sourceCode: string): ResultPtr;
  run_nano(sourceCode: string): ResultPtr;
  free_result(ptr: ResultPtr): void;
}

export class NanolangWASM {
  private module: NanolangModule;
  
  async init() {
    this.module = await loadNanolangWASM();
  }
  
  compile(sourceCode: string): CompileResult {
    const resultPtr = this.module.compile_nano(sourceCode);
    const result = this.readResult(resultPtr);
    this.module.free_result(resultPtr);
    return result;
  }
  
  run(sourceCode: string): ExecutionResult {
    const resultPtr = this.module.run_nano(sourceCode);
    const result = this.readResult(resultPtr);
    this.module.free_result(resultPtr);
    return result;
  }
  
  private readResult(ptr: ResultPtr): any {
    // Read memory at pointer to extract result
    // ...
  }
}
```

## Backend Services

### Code Sharing Service

```typescript
// backend/src/routes/share.ts

app.post('/api/share', async (req, res) => {
  const { code, settings } = req.body;
  
  // Generate short ID
  const id = generateShortId();
  
  // Store in database
  await db.saveCode({
    id,
    code,
    settings,
    created_at: new Date()
  });
  
  // Return shareable URL
  res.json({
    url: `https://play.nanolang.org/${id}`
  });
});

app.get('/api/code/:id', async (req, res) => {
  const { id } = req.params;
  
  const code = await db.getCode(id);
  if (!code) {
    return res.status(404).json({ error: 'Code not found' });
  }
  
  res.json(code);
});
```

### Example Gallery

```typescript
// backend/src/routes/examples.ts

const examples = [
  {
    id: 'hello',
    title: 'Hello World',
    description: 'Print a greeting',
    code: 'fn main() -> int {\n  (println "Hello, World!")\n  return 0\n}',
    category: 'basics'
  },
  {
    id: 'factorial',
    title: 'Factorial',
    description: 'Recursive factorial function',
    code: '...',
    category: 'recursion'
  },
  // ... more examples
];

app.get('/api/examples', (req, res) => {
  res.json(examples);
});

app.get('/api/examples/:id', (req, res) => {
  const example = examples.find(e => e.id === req.params.id);
  if (!example) {
    return res.status(404).json({ error: 'Example not found' });
  }
  res.json(example);
});
```

## Features

### Phase 1: Core Functionality
- [x] Code editor with syntax highlighting
- [x] Compile button
- [x] Run button (interpreter mode)
- [x] Error display
- [x] Output console
- [x] Example selection

### Phase 2: Enhanced Features
- [ ] Share code via URL
- [ ] Embed playground in external sites
- [ ] Download compiled code
- [ ] Format code
- [ ] Keyboard shortcuts
- [ ] Mobile-responsive design

### Phase 3: Advanced Features
- [ ] Multi-file projects
- [ ] Import external modules
- [ ] Debugging (step through execution)
- [ ] Performance profiling
- [ ] Collaborative editing (real-time)
- [ ] Version history

## UI/UX Design

### Layout

```
┌─────────────────────────────────────────┐
│  Nanolang Playground    [Run] [Share]   │
├──────────────────┬──────────────────────┤
│                  │                      │
│                  │    Output Console    │
│   Code Editor    │    ┌──────────────┐ │
│                  │    │              │ │
│                  │    │ > Compiling..│ │
│                  │    │ > Success!   │ │
│                  │    │ > 42         │ │
│                  │    │              │ │
│                  │    └──────────────┘ │
│                  │                      │
│                  │    [ Examples ▾ ]    │
├──────────────────┴──────────────────────┤
│  [Docs] [Tutorial] [Feedback]           │
└─────────────────────────────────────────┘
```

### Color Scheme

```css
:root {
  --bg-primary: #1e1e1e;
  --bg-secondary: #252526;
  --text-primary: #d4d4d4;
  --text-secondary: #858585;
  --accent: #007acc;
  --success: #4ec9b0;
  --error: #f48771;
}
```

## Technology Stack

### Frontend
- **React**: UI framework
- **Monaco Editor**: Code editor
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **React Query**: Data fetching

### Backend
- **Node.js + Express**: API server
- **PostgreSQL**: Database for shared code
- **Redis**: Caching
- **nginx**: Reverse proxy

### Infrastructure
- **Vercel/Netlify**: Frontend hosting
- **Heroku/Railway**: Backend hosting
- **CDN**: Static asset delivery

## Performance Optimization

### WASM Optimization

```bash
# Optimize for size and speed
emcc -O3 -s WASM=1 \
  -s AGGRESSIVE_VARIABLE_ELIMINATION=1 \
  -s ELIMINATE_DUPLICATE_FUNCTIONS=1 \
  --closure 1 \
  src/*.c -o public/nanolang.js
```

### Lazy Loading

```typescript
// Lazy load WASM module
const NanolangWASM = React.lazy(() => import('./wasm/nanolang'));

// Show spinner while loading
<Suspense fallback={<LoadingSpinner />}>
  <NanolangWASM />
</Suspense>
```

### Code Splitting

```typescript
// Split by route
const Examples = React.lazy(() => import('./pages/Examples'));
const Docs = React.lazy(() => import('./pages/Docs'));
```

## Implementation Roadmap

### Milestone 1: MVP (4 weeks)
- Basic editor with syntax highlighting
- WASM compiler integration
- Run button with output display
- Error messages

### Milestone 2: Sharing (2 weeks)
- Backend API for code storage
- Share button and URL generation
- Load shared code

### Milestone 3: Examples (1 week)
- Example gallery
- Load examples into editor
- Categorization

### Milestone 4: Polish (2 weeks)
- Mobile responsiveness
- Dark/light themes
- Settings panel
- Keyboard shortcuts

### Milestone 5: Advanced (4 weeks)
- Multi-file support
- Module imports
- Debugging features
- Performance profiling

## Testing

### Unit Tests
```typescript
describe('NanolangWASM', () => {
  it('compiles valid code', async () => {
    const wasm = new NanolangWASM();
    await wasm.init();
    
    const result = wasm.compile('fn main() -> int { return 42 }');
    expect(result.exitCode).toBe(0);
    expect(result.errors).toBe('');
  });
  
  it('reports syntax errors', async () => {
    const wasm = new NanolangWASM();
    await wasm.init();
    
    const result = wasm.compile('fn main() { invalid }');
    expect(result.exitCode).not.toBe(0);
    expect(result.errors).toContain('syntax error');
  });
});
```

### E2E Tests
```typescript
describe('Playground', () => {
  it('compiles and runs code', () => {
    cy.visit('/');
    cy.get('.editor').type('fn main() -> int { (println "Hello") return 0 }');
    cy.get('[data-testid="run-button"]').click();
    cy.get('.console-output').should('contain', 'Hello');
  });
  
  it('shares code', () => {
    cy.visit('/');
    cy.get('.editor').type('fn main() -> int { return 42 }');
    cy.get('[data-testid="share-button"]').click();
    cy.get('.share-url').should('match', /https:\/\/play\.nanolang\.org\/\w+/);
  });
});
```

## Future Enhancements

1. **AI Assistant**: Code completion and suggestions via GPT
2. **Visualizations**: Graphical representation of data structures
3. **Jupyter Integration**: Nanolang kernel for Jupyter notebooks
4. **Classroom Mode**: Teacher dashboard for assignments
5. **Competitive Programming**: Timed challenges and leaderboards

## References

- [Rust Playground](https://play.rust-lang.org/)
- [TypeScript Playground](https://www.typescriptlang.org/play)
- [Go Playground](https://go.dev/play/)
- [Emscripten Documentation](https://emscripten.org/)

## Related Issues

- WASM compilation target
- Web-based REPL (subset of playground)
- Online documentation integration
- Tutorial system

