"use strict";
// Compiled output — source: src/extension.ts
// nano-fmt format-on-save + LSP client + semantic tokens
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) { o["default"] = v; });
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;

const vscode = __importStar(require("vscode"));
const cp = __importStar(require("child_process"));
const path = __importStar(require("path"));
const node_1 = require("vscode-languageclient/node");

let client;
let fmtWarnShown = false;

// ---------------------------------------------------------------------------
// Format-on-save via nano-fmt --stdin
// ---------------------------------------------------------------------------
async function formatWithNanoFmt(document) {
    const config = vscode.workspace.getConfiguration("nanolang");
    if (!config.get("formatOnSave", true)) return [];
    const fmtPath = config.get("fmtPath", "nano-fmt");
    const original = document.getText();
    return new Promise((resolve) => {
        const proc = cp.spawn(fmtPath, ["--stdin"], {
            cwd: path.dirname(document.uri.fsPath),
        });
        let stdout = "";
        let stderr = "";
        proc.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
        proc.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
        proc.on("error", (_err) => {
            if (!fmtWarnShown) {
                fmtWarnShown = true;
                vscode.window.showWarningMessage(
                    `nano-fmt not found at '${fmtPath}'. Install it or set nanolang.fmtPath.`
                );
            }
            resolve([]);
        });
        proc.on("close", (code) => {
            if (code !== 0 || stderr) {
                if (stderr) vscode.window.setStatusBarMessage(`nano-fmt: ${stderr.slice(0, 80)}`, 3000);
                resolve([]);
                return;
            }
            if (stdout === original) { resolve([]); return; }
            const fullRange = new vscode.Range(
                document.positionAt(0),
                document.positionAt(original.length)
            );
            resolve([vscode.TextEdit.replace(fullRange, stdout)]);
        });
        proc.stdin.write(original);
        proc.stdin.end();
    });
}

// ---------------------------------------------------------------------------
// activate
// ---------------------------------------------------------------------------
function activate(context) {
    const config = vscode.workspace.getConfiguration("nanolang");
    const serverExecutable = config.get("lspPath", "nanolang-lsp");

    // ── LSP client ──────────────────────────────────────────────────────────
    const serverOptions = {
        command: serverExecutable,
        transport: node_1.TransportKind.stdio,
    };
    const clientOptions = {
        documentSelector: [{ scheme: "file", language: "nanolang" }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher("**/*.nano"),
        },
        // Semantic tokens are provided by the server; package.json maps
        // token types to TextMate scopes via semanticTokenScopes.
        middleware: {},
    };
    client = new node_1.LanguageClient(
        "nanolang", "Nanolang Language Server", serverOptions, clientOptions
    );
    client.start().catch((err) => {
        vscode.window.showErrorMessage(`nanolang-lsp failed to start: ${err.message}`);
    });

    // ── Format-on-save provider ─────────────────────────────────────────────
    const formattingProvider = vscode.languages.registerDocumentFormattingEditProvider(
        { language: "nanolang", scheme: "file" },
        {
            provideDocumentFormattingEdits(document) {
                return formatWithNanoFmt(document);
            },
        }
    );
    context.subscriptions.push(formattingProvider);

    // ── Status bar ──────────────────────────────────────────────────────────
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = "$(symbol-misc) nano";
    statusBar.tooltip = "Nanolang LSP active";
    statusBar.show();
    context.subscriptions.push(statusBar);

    // ── Command: nanolang.format ────────────────────────────────────────────
    const formatCmd = vscode.commands.registerCommand("nanolang.format", async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== "nanolang") return;
        const edits = await formatWithNanoFmt(editor.document);
        if (edits.length > 0) {
            const wsEdit = new vscode.WorkspaceEdit();
            wsEdit.set(editor.document.uri, edits);
            await vscode.workspace.applyEdit(wsEdit);
            vscode.window.setStatusBarMessage("nano-fmt: formatted", 2000);
        } else {
            vscode.window.setStatusBarMessage("nano-fmt: already formatted", 1500);
        }
    });
    context.subscriptions.push(formatCmd);
}

function deactivate() {
    if (client) return client.stop();
    return undefined;
}
