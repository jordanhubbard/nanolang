import * as vscode from "vscode";
import {
  LanguageClient,
  ServerOptions,
  LanguageClientOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration("nanolang");
  const serverExecutable = config.get<string>("lspPath", "nanolang-lsp");

  const serverOptions: ServerOptions = {
    command: serverExecutable,
    transport: TransportKind.stdio,
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "nanolang" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.nano"),
    },
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
}

export function deactivate(): Thenable<void> | undefined {
  if (client) {
    return client.stop();
  }
  return undefined;
}
