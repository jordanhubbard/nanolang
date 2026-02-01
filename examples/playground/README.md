# 🔬 NanoLang Playground

**Interactive web-based development environment for NanoLang**

Inspired by Swift Playgrounds, the NanoLang Playground provides an interactive way to learn, explore, and experiment with NanoLang directly in your web browser.

## ✨ Features

- **📝 Code Editor** - Syntax-aware editor with monospace font and line numbers
- **📚 Example Gallery** - 10+ curated examples covering key language features
- **▶️ Code Execution** - Compile and run NanoLang code in the browser
- **📋 Copy & Download** - Export your code for local compilation
- **⌨️ Keyboard Shortcuts** - Ctrl+Enter to run, Ctrl+S to download
- **📱 Responsive Design** - Works on desktop, tablet, and mobile
- **🎨 Beautiful UI** - Modern, clean interface inspired by Swift Playgrounds

## 🚀 Quick Start

### Build and Run

```bash
# From the nanolang repository root:

# 1. Compile the playground server
./bin/nanoc examples/playground/playground_server.nano -o bin/playground

# 2. Start the server
./bin/playground

# 3. Open your browser
open http://localhost:8080
```

### Alternative: Simple Static Server

If you prefer to use a simple HTTP server without compiling:

```bash
# Using Python
cd examples/playground/public
python3 -m http.server 8080

# Or using Node.js
npx http-server -p 8080
```

Then open `http://localhost:8080` in your browser.

## 📖 Examples Included

The playground comes with 10 interactive examples:

1. **Hello World** - Basic function definitions and shadow tests
2. **Factorial** - Recursive factorial calculation
3. **Fibonacci** - Fibonacci sequence generation
4. **Prime Numbers** - Prime number checker with loops
5. **Array Operations** - Array manipulation (sum, max, etc.)
6. **String Manipulation** - String operations and character handling
7. **Structs** - Defining and using struct types
8. **Conditionals (cond)** - Multi-way branching with cond expressions
9. **Loops** - while and for loop patterns
10. **Recursion** - Various recursive algorithms

## 🎮 How to Use

### Loading Examples

1. Click any example in the sidebar
2. The code appears in the editor
3. Click "Run Code" or press **Ctrl+Enter** to validate
4. View output and any warnings/errors

### Writing Your Own Code

1. Click "Clear" to start with a blank editor
2. Write your NanoLang code
3. Include shadow tests for all functions
4. Run to validate syntax

### Exporting Code

- **Copy to Clipboard**: Click the 📋 button
- **Download as File**: Click the 💾 button or press **Ctrl+S**

### Code Execution

When running with the NanoLang playground server, your code is compiled and executed on the server:

1. Click "Run Code" or press **Ctrl+Enter**
2. The code is sent to the server's `/api/execute` endpoint
3. The server compiles and runs your code
4. Output (or errors) are displayed in real-time

**Note:** If the server is unavailable (e.g., using a simple static file server), the playground falls back to client-side syntax validation only.

## 🏗️ Architecture

### Frontend (Static HTML/JS/CSS)

- **index.html** - Main page structure
- **app.js** - Application logic and UI interactions
- **examples.js** - Example code library
- **style.css** - Visual styling

### Backend (NanoLang HTTP Server)

- **playground_server.nano** - HTTP server serving static files
- Uses `http_server` module with built-in `/api/execute` endpoint
- Server compiles and executes submitted code, returning JSON results
- Set `PLAYGROUND_PORT` environment variable to change the port (default: 8080)

### Execution Flow

1. User writes code in the browser editor
2. Code is sent to `/api/execute` via POST
3. Server writes code to temp file, compiles with `nanoc`
4. If compilation succeeds, runs the binary and captures output
5. Returns JSON with success/failure, compile output, and program output

## 🎯 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` (or `Cmd+Enter`) | Run code |
| `Ctrl+S` (or `Cmd+S`) | Download code |

## 🔧 Customization

### Adding New Examples

Edit `public/examples.js` and add to the `EXAMPLES` object:

```javascript
myexample: {
    title: "My Example",
    description: "What this example demonstrates",
    code: `fn my_function() -> int {
    return 42
}

shadow my_function {
    assert (== (my_function) 42)
}`
}
```

Then add a button in `index.html`:

```html
<button class="example-btn" data-example="myexample">My Example</button>
```

### Styling

Modify `public/style.css` to customize:
- Colors (CSS variables in `:root`)
- Layout (grid configuration)
- Fonts
- Animations

## 🚧 Future Enhancements
- [ ] **Syntax Highlighting** - Full syntax highlighting with CodeMirror/Monaco
- [ ] **Auto-completion** - IntelliSense-style code completion
- [ ] **Share Links** - Share code via URLs
- [ ] **Mobile App** - Native iOS/Android apps
- [ ] **Collaborative Editing** - Real-time collaboration
- [ ] **REPL Mode** - Interactive line-by-line execution
- [ ] **Debugger** - Step-through debugging
- [ ] **Performance Profiling** - Timing and memory analysis
- [ ] **Package Explorer** - Browse standard library and modules

## 📚 Learning Resources

- [NanoLang Documentation](https://jordanhubbard.github.io/nanolang/)
- [Quick Reference](../../docs/QUICK_REFERENCE.md)
- [Getting Started Guide](../../docs/GETTING_STARTED.md)
- [Language Specification](../../docs/SPECIFICATION.md)

## 🤝 Contributing

Contributions welcome! Areas to improve:

1. **More Examples** - Add examples for advanced features (unions, generics, FFI)
2. **Better Validation** - Improve client-side syntax checking
3. **UI Enhancements** - Better error messages, syntax highlighting
4. **Sandboxed Execution** - Add sandboxing for safer code execution
5. **Mobile Optimization** - Improve mobile user experience

## 📄 License

Apache License 2.0 - See [LICENSE](../../LICENSE) for details

## 🙏 Acknowledgments

- Inspired by [Swift Playgrounds](https://www.apple.com/swift/playgrounds/)
- Built with vanilla JavaScript (no frameworks!)
- Uses the NanoLang `http_server` module

---

**Happy Coding! 🎉**

For questions or feedback, open an issue on [GitHub](https://github.com/jordanhubbard/nanolang/issues).
