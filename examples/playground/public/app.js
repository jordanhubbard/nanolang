// NanoLang Playground — app.js
// Runs NanoLang directly in the browser via the Emscripten-compiled interpreter.

// ── WASM module ───────────────────────────────────────────────────────────────
let NL = null;          // createNanolang() result
let nl_run   = null;    // cwrap'd nl_run(source) → int
let nl_check = null;    // cwrap'd nl_check(source) → int
let nl_ver   = null;    // cwrap'd nl_version() → string

// Mutable print handlers — swapped per-run via the stable closures below.
// Emscripten binds its internal `out`/`err` once at init, so we must
// use an indirection layer rather than patching NL.print directly.
let _printHandler    = s => console.log(s);
let _printErrHandler = s => console.warn(s);

const wasmReady = createNanolang({
    // Stable closures — these are captured once by Emscripten at init.
    print:    s => _printHandler(s),
    printErr: s => _printErrHandler(s),
    noInitialRun: true,
    locateFile: (path) => path,   // serve from same directory
}).then(module => {
    NL       = module;
    nl_run   = NL.cwrap('nl_run',    'number', ['string']);
    nl_check = NL.cwrap('nl_check',  'number', ['string']);
    nl_ver   = NL.cwrap('nl_version','string', []);
    const ver = nl_ver();
    document.querySelectorAll('.nl-version').forEach(el => el.textContent = ver);
    setStatus('ready', `NanoLang ${ver} loaded — runs in browser`);
    runBtn.disabled = false;
}).catch(err => {
    console.error('WASM load failed:', err);
    setStatus('error', 'Failed to load NanoLang WASM — server fallback active');
});

// ── ANSI escape-code stripper ─────────────────────────────────────────────────
const ANSI_RE = /\x1b\[[0-9;]*[A-Za-z]/g;
function stripAnsi(s) { return s.replace(ANSI_RE, ''); }

// ── State ─────────────────────────────────────────────────────────────────────
let currentExample = null;
let isDirty = false;

// ── DOM ───────────────────────────────────────────────────────────────────────
const codeEditor    = document.getElementById('code-editor');
const outputDiv     = document.getElementById('output');
const errorsDiv     = document.getElementById('errors');
const runBtn        = document.getElementById('run-btn');
const clearBtn      = document.getElementById('clear-btn');
const clearOutputBtn= document.getElementById('clear-output-btn');
const copyBtn       = document.getElementById('copy-btn');
const downloadBtn   = document.getElementById('download-btn');
const statusBar     = document.getElementById('status-bar');

runBtn.disabled = true;  // enabled once WASM is ready

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadExample('hello');
    setupEventListeners();
});

// ── Event wiring ──────────────────────────────────────────────────────────────
function setupEventListeners() {
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => loadExample(btn.getAttribute('data-example')));
    });

    runBtn.addEventListener('click', runCode);

    clearBtn.addEventListener('click', () => {
        if (confirm('Clear the editor? This cannot be undone.')) {
            codeEditor.value = '';
            clearOutput();
            isDirty = false;
        }
    });

    clearOutputBtn.addEventListener('click', clearOutput);
    copyBtn.addEventListener('click', copyToClipboard);
    downloadBtn.addEventListener('click', downloadCode);

    codeEditor.addEventListener('input', () => { isDirty = true; });

    document.addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); runCode(); }
        if ((e.ctrlKey || e.metaKey) && e.key === 's')     { e.preventDefault(); downloadCode(); }
    });
}

// ── Example loader ────────────────────────────────────────────────────────────
function loadExample(key) {
    const example = EXAMPLES[key];
    if (!example) return;
    if (isDirty && !confirm('You have unsaved changes. Load example anyway?')) return;

    codeEditor.value = example.code;
    currentExample = key;
    isDirty = false;
    clearOutput();

    outputDiv.innerHTML = `<div class="info-message">
        <strong>${escapeHtml(example.title)}</strong><br>
        ${escapeHtml(example.description)}<br><br>
        Press ▶ Run Code or Ctrl+Enter to execute in browser.
    </div>`;

    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-example') === key);
    });
}

// ── Core: run code ────────────────────────────────────────────────────────────
async function runCode() {
    const code = codeEditor.value.trim();
    if (!code) { showError('No code to run. Please write some NanoLang code.'); return; }

    clearOutput();
    outputDiv.innerHTML = '<div class="info-message">⏳ Running in browser…</div>';
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Running…';
    setStatus('running', 'Executing…');

    await wasmReady.catch(() => {});   // wait for WASM even if called early

    if (NL) {
        runWithWasm(code);
    } else {
        runWithServer(code);
    }
}

function runWithWasm(code) {
    let stdout = '';
    let stderr = '';

    // Swap in capture handlers via the indirection layer set up at init time
    const savedPrint   = _printHandler;
    const savedPrintErr= _printErrHandler;
    _printHandler    = line => { stdout += line + '\n'; };
    _printErrHandler = line => { stderr += line + '\n'; };

    let rc;
    try {
        rc = nl_run(code);
    } finally {
        _printHandler    = savedPrint;
        _printErrHandler = savedPrintErr;
    }

    stdout = stripAnsi(stdout);
    stderr = stripAnsi(stderr);

    runBtn.disabled = false;
    runBtn.textContent = '▶ Run Code';

    if (rc === 0) {
        let html = '<div class="success-message">✅ <strong>Ran successfully (exit 0)</strong></div>';
        if (stdout.trim()) {
            html += '<div class="program-output"><strong>Output:</strong><pre>' + escapeHtml(stdout) + '</pre></div>';
        } else {
            html += '<div class="info-message">Program produced no output.</div>';
        }
        outputDiv.innerHTML = html;
        errorsDiv.innerHTML = stderr.trim()
            ? '<div class="warning-message"><strong>Warnings/diagnostics:</strong><pre>' + escapeHtml(stderr) + '</pre></div>'
            : '<div class="success-message">No errors or warnings.</div>';
        setStatus('ready', 'Done');
    } else {
        outputDiv.innerHTML = '<div class="error-message">❌ <strong>Program exited with code ' + rc + '</strong></div>';
        let errHtml = '';
        if (stderr.trim()) {
            errHtml += '<div class="error-details"><pre>' + escapeHtml(stderr) + '</pre></div>';
        }
        if (stdout.trim()) {
            errHtml += '<div class="compile-output"><strong>Output before error:</strong><pre>' + escapeHtml(stdout) + '</pre></div>';
        }
        errorsDiv.innerHTML = errHtml || '<div class="error-message">No diagnostic output.</div>';
        setStatus('error', 'Error (exit ' + rc + ')');
    }
}

// ── Server fallback (for programs that need OS resources) ─────────────────────
function runWithServer(code) {
    fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: code,
    })
    .then(r => r.json())
    .then(result => {
        runBtn.disabled = false;
        runBtn.textContent = '▶ Run Code';
        displayServerResult(result);
    })
    .catch(err => {
        runBtn.disabled = false;
        runBtn.textContent = '▶ Run Code';
        outputDiv.innerHTML = '<div class="error-message">❌ Server unavailable and WASM not loaded.</div>';
        setStatus('error', 'Unavailable');
    });
}

function displayServerResult(result) {
    if (result.success) {
        let html = '<div class="success-message">✅ <strong>Execution successful (server)</strong></div>';
        if (result.output?.trim())
            html += '<div class="program-output"><strong>Output:</strong><pre>' + escapeHtml(result.output) + '</pre></div>';
        outputDiv.innerHTML = html;
        errorsDiv.innerHTML = '<div class="success-message">No errors.</div>';
        setStatus('ready', 'Done (server)');
    } else {
        let errHtml = '<div class="error-message">❌ <strong>Execution failed (server)</strong></div>';
        if (result.error) errHtml += '<div class="error-details">' + escapeHtml(result.error) + '</div>';
        if (result.output?.trim())
            errHtml += '<pre>' + escapeHtml(result.output) + '</pre>';
        errorsDiv.innerHTML = errHtml;
        outputDiv.innerHTML = '<div class="error-message">Please fix the errors and try again.</div>';
        setStatus('error', 'Error');
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function clearOutput() {
    outputDiv.innerHTML = '';
    errorsDiv.innerHTML = '';
}

function showError(msg) {
    errorsDiv.innerHTML = `<div class="error-message">❌ ${escapeHtml(msg)}</div>`;
}

function setStatus(state, msg) {
    if (!statusBar) return;
    statusBar.textContent = msg;
    statusBar.className = 'status-bar status-' + state;
}

function copyToClipboard() {
    navigator.clipboard.writeText(codeEditor.value).then(() => {
        const orig = copyBtn.textContent;
        copyBtn.textContent = '✅';
        setTimeout(() => { copyBtn.textContent = orig; }, 1500);
    }).catch(() => showError('Failed to copy to clipboard'));
}

function downloadCode() {
    const code = codeEditor.value;
    const filename = currentExample ? `${currentExample}.nano` : 'playground.nano';
    const blob = new Blob([code], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    const orig = downloadBtn.textContent;
    downloadBtn.textContent = '✅';
    setTimeout(() => { downloadBtn.textContent = orig; }, 1500);
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
