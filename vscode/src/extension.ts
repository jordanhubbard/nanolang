import * as vscode from "vscode";
import * as cp from "child_process";
import * as path from "path";
import {
  LanguageClient,
  ServerOptions,
  LanguageClientOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient;

// ---------------------------------------------------------------------------
// Format-on-save via nano-fmt
// nano-fmt reads stdin and writes formatted output to stdout (--stdin flag).
// Falls back to no-op if nano-fmt is not found, showing a one-time warning.
// ---------------------------------------------------------------------------
let fmtWarnShown = false;

async function formatWithNanoFmt(
  document: vscode.TextDocument
): Promise<vscode.TextEdit[]> {
  const config = vscode.workspace.getConfiguration("nanolang");
  if (!config.get<boolean>("formatOnSave", true)) return [];

  const fmtPath = config.get<string>("fmtPath", "nano-fmt");
  const original = document.getText();

  return new Promise((resolve) => {
    const proc = cp.spawn(fmtPath, ["--stdin"], {
      cwd: path.dirname(document.uri.fsPath),
    });

    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
    proc.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
    proc.on("error", (err: NodeJS.ErrnoException) => {
      if (!fmtWarnShown) {
        fmtWarnShown = true;
        vscode.window.showWarningMessage(
          `nano-fmt not found at '${fmtPath}'. Install it or set nanolang.fmtPath.`
        );
      }
      resolve([]);
    });
    proc.on("close", (code: number | null) => {
      if (code !== 0 || stderr) {
        // Don't replace document on formatter error — show status
        if (stderr) {
          vscode.window.setStatusBarMessage(`nano-fmt: ${stderr.slice(0, 80)}`, 3000);
        }
        resolve([]);
        return;
      }
      if (stdout === original) {
        resolve([]); // no change
        return;
      }
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
// Extension activate
// ---------------------------------------------------------------------------
export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration("nanolang");
  const serverExecutable = config.get<string>("lspPath", "nanolang-lsp");

  // ── LSP client ────────────────────────────────────────────────────────────
  const serverOptions: ServerOptions = {
    command: serverExecutable,
    transport: TransportKind.stdio,
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "nanolang" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.nano"),
    },
    // Enable semantic tokens: the LSP server advertises semanticTokensProvider
    // with full-document encoding; VS Code uses semanticTokenScopes from
    // package.json to map token types to TextMate scopes for theming.
    middleware: {},
  };

  client = new LanguageClient(
    "nanolang",
    "Nanolang Language Server",
    serverOptions,
    clientOptions
  );

  client.start().catch((err: Error) => {
    vscode.window.showErrorMessage(
      `nanolang-lsp failed to start: ${err.message}`
    );
  });

  // ── Format-on-save document formatting provider ───────────────────────────
  // VS Code calls this provider when the user saves with editor.formatOnSave
  // or when nanolang.formatOnSave is enabled.
  const formattingProvider = vscode.languages.registerDocumentFormattingEditProvider(
    { language: "nanolang", scheme: "file" },
    {
      provideDocumentFormattingEdits(
        document: vscode.TextDocument
      ): Promise<vscode.TextEdit[]> {
        return formatWithNanoFmt(document);
      },
    }
  );
  context.subscriptions.push(formattingProvider);

  // ── Status bar item ───────────────────────────────────────────────────────
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.text = "$(symbol-misc) nano";
  statusBar.tooltip = "Nanolang LSP active";
  statusBar.show();
  context.subscriptions.push(statusBar);

  // ── Command: force format ─────────────────────────────────────────────────
  const formatCmd = vscode.commands.registerCommand(
    "nanolang.format",
    async () => {
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
    }
  );
  context.subscriptions.push(formatCmd);
}

export function deactivate(): Thenable<void> | undefined {
  if (client) {
    return client.stop();
  }
  return undefined;
}
