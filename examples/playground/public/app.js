// NanoLang Playground Application Logic

// State
let currentExample = null;
let isDirty = false;

// DOM Elements
const codeEditor = document.getElementById('code-editor');
const outputDiv = document.getElementById('output');
const errorsDiv = document.getElementById('errors');
const runBtn = document.getElementById('run-btn');
const clearBtn = document.getElementById('clear-btn');
const clearOutputBtn = document.getElementById('clear-output-btn');
const copyBtn = document.getElementById('copy-btn');
const downloadBtn = document.getElementById('download-btn');
const examplesList = document.getElementById('examples-list');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Load default example
    loadExample('hello');

    // Setup event listeners
    setupEventListeners();

    // Add syntax highlighting on load
    highlightSyntax();
});

function setupEventListeners() {
    // Example buttons
    const exampleButtons = document.querySelectorAll('.example-btn');
    exampleButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const exampleKey = this.getAttribute('data-example');
            loadExample(exampleKey);
        });
    });

    // Run button
    runBtn.addEventListener('click', runCode);

    // Clear button
    clearBtn.addEventListener('click', function() {
        if (confirm('Clear the editor? This cannot be undone.')) {
            codeEditor.value = '';
            clearOutput();
            isDirty = false;
        }
    });

    // Clear output button
    clearOutputBtn.addEventListener('click', clearOutput);

    // Copy button
    copyBtn.addEventListener('click', copyToClipboard);

    // Download button
    downloadBtn.addEventListener('click', downloadCode);

    // Track changes
    codeEditor.addEventListener('input', function() {
        isDirty = true;
    });

    // Auto-resize textarea
    codeEditor.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to run
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            runCode();
        }
        // Ctrl/Cmd + S to save/download
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            downloadCode();
        }
    });
}

function loadExample(key) {
    const example = EXAMPLES[key];
    if (!example) {
        console.error('Example not found:', key);
        return;
    }

    // Check if editor has unsaved changes
    if (isDirty && !confirm('You have unsaved changes. Load example anyway?')) {
        return;
    }

    codeEditor.value = example.code;
    currentExample = key;
    isDirty = false;

    // Clear output
    clearOutput();

    // Show example info
    outputDiv.innerHTML = `<div class="info-message">
        <strong>${example.title}</strong><br>
        ${example.description}<br><br>
        Press "Run Code" or Ctrl+Enter to validate syntax
    </div>`;

    // Highlight active example button
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-example') === key) {
            btn.classList.add('active');
        }
    });
}

function runCode() {
    const code = codeEditor.value.trim();

    if (!code) {
        showError('No code to run. Please write some NanoLang code.');
        return;
    }

    clearOutput();

    // Show running state
    outputDiv.innerHTML = '<div class="info-message">⏳ Compiling and running...</div>';
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Running...';

    // Call the execute API
    fetch('/api/execute', {
        method: 'POST',
        headers: {
            'Content-Type': 'text/plain'
        },
        body: code
    })
    .then(response => response.json())
    .then(result => {
        runBtn.disabled = false;
        runBtn.textContent = '▶️ Run Code';
        displayExecutionResult(result);
    })
    .catch(error => {
        runBtn.disabled = false;
        runBtn.textContent = '▶️ Run Code';
        // Fall back to client-side validation if server unavailable
        console.warn('Server unavailable, falling back to syntax validation:', error);
        outputDiv.innerHTML = '<div class="warning-message">⚠️ Server unavailable. Performing syntax validation only...</div>';
        setTimeout(() => validateSyntax(code), 300);
    });
}

function displayExecutionResult(result) {
    if (result.success) {
        let output = '<div class="success-message">✅ <strong>Execution successful!</strong></div>';
        
        if (result.compile_output && result.compile_output.trim()) {
            output += '<div class="compile-output"><strong>Compile output:</strong><pre>' + 
                      escapeHtml(result.compile_output) + '</pre></div>';
        }
        
        if (result.output && result.output.trim()) {
            output += '<div class="program-output"><strong>Program output:</strong><pre>' + 
                      escapeHtml(result.output) + '</pre></div>';
        } else {
            output += '<div class="info-message">Program produced no output.</div>';
        }
        
        outputDiv.innerHTML = output;
        errorsDiv.innerHTML = '<div class="success-message">No errors</div>';
    } else {
        let errorOutput = '<div class="error-message">❌ <strong>Execution failed</strong></div>';
        
        if (result.error) {
            errorOutput += '<div class="error-details"><strong>Error:</strong> ' + 
                          escapeHtml(result.error) + '</div>';
        }
        
        if (result.output && result.output.trim()) {
            errorOutput += '<div class="compile-output"><strong>Output:</strong><pre>' + 
                          escapeHtml(result.output) + '</pre></div>';
        }
        
        errorsDiv.innerHTML = errorOutput;
        outputDiv.innerHTML = '<div class="error-message">Please fix the errors and try again.</div>';
    }
}

function validateSyntax(code) {
    const errors = [];
    const warnings = [];

    // Basic syntax checks
    if (!code.includes('fn main')) {
        warnings.push({
            line: 0,
            message: 'No main() function found. Add: fn main() -> int { return 0 }'
        });
    }

    // Check for shadow tests
    const functionMatches = code.match(/fn\s+(\w+)/g);
    if (functionMatches) {
        functionMatches.forEach(match => {
            const fnName = match.split(/\s+/)[1];
            if (fnName !== 'main' && !code.includes(`shadow ${fnName}`)) {
                warnings.push({
                    line: 0,
                    message: `Function '${fnName}' is missing a shadow test`
                });
            }
        });
    }

    // Check for balanced braces
    let braceCount = 0;
    let parenCount = 0;
    for (let char of code) {
        if (char === '{') braceCount++;
        if (char === '}') braceCount--;
        if (char === '(') parenCount++;
        if (char === ')') parenCount--;
    }

    if (braceCount !== 0) {
        errors.push({
            line: 0,
            message: 'Unbalanced braces: ' + (braceCount > 0 ? 'missing }' : 'extra }')
        });
    }

    if (parenCount !== 0) {
        errors.push({
            line: 0,
            message: 'Unbalanced parentheses: ' + (parenCount > 0 ? 'missing )' : 'extra )')
        });
    }

    // Display results
    if (errors.length === 0) {
        let output = '<div class="success-message">✅ <strong>Syntax validation passed!</strong><br><br>';
        output += 'Your code appears to be valid NanoLang syntax.<br><br>';
        output += '<strong>To execute this code:</strong><br>';
        output += '1. Copy the code (📋 button)<br>';
        output += '2. Save it to a .nano file<br>';
        output += '3. Compile: <code>./bin/nanoc yourfile.nano -o output</code><br>';
        output += '4. Run: <code>./output</code></div>';

        outputDiv.innerHTML = output;

        if (warnings.length > 0) {
            let warningHtml = '<div class="warning-message"><strong>⚠️ Warnings:</strong><ul>';
            warnings.forEach(w => {
                warningHtml += `<li>${w.message}</li>`;
            });
            warningHtml += '</ul></div>';
            errorsDiv.innerHTML = warningHtml;
        } else {
            errorsDiv.innerHTML = '<div class="success-message">No warnings or errors</div>';
        }
    } else {
        let errorHtml = '<div class="error-message"><strong>❌ Syntax Errors:</strong><ul>';
        errors.forEach(e => {
            errorHtml += `<li>${e.message}</li>`;
        });
        errorHtml += '</ul></div>';

        errorsDiv.innerHTML = errorHtml;
        outputDiv.innerHTML = '<div class="error-message">Please fix syntax errors before running</div>';
    }
}

function clearOutput() {
    outputDiv.innerHTML = '';
    errorsDiv.innerHTML = '';
}

function showError(message) {
    errorsDiv.innerHTML = `<div class="error-message">❌ ${message}</div>`;
}

function copyToClipboard() {
    const code = codeEditor.value;
    navigator.clipboard.writeText(code).then(() => {
        // Show feedback
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '✅';
        setTimeout(() => {
            copyBtn.textContent = originalText;
        }, 1500);
    }).catch(err => {
        showError('Failed to copy to clipboard');
    });
}

function downloadCode() {
    const code = codeEditor.value;
    const filename = currentExample ? `${currentExample}.nano` : 'playground.nano';

    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Show feedback
    const originalText = downloadBtn.textContent;
    downloadBtn.textContent = '✅';
    setTimeout(() => {
        downloadBtn.textContent = originalText;
    }, 1500);
}

function highlightSyntax() {
    // Simple syntax highlighting (basic version)
    // In a production version, you'd use a proper syntax highlighter
    // like CodeMirror or Monaco Editor
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
