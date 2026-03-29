'use strict';

const { workspace, window } = require('vscode');
const {
  LanguageClient,
  TransportKind
} = require('vscode-languageclient/node');

let client;

function activate(context) {
  const config = workspace.getConfiguration('nanolang');
  const serverPath = config.get('serverPath') || 'nanolang-lsp';

  const serverOptions = {
    command: serverPath,
    transport: TransportKind.stdio
  };

  const clientOptions = {
    documentSelector: [{ scheme: 'file', language: 'nanolang' }],
    synchronize: {
      fileEvents: workspace.createFileSystemWatcher('**/*.nano')
    }
  };

  client = new LanguageClient(
    'nanolang',
    'NanoLang Language Server',
    serverOptions,
    clientOptions
  );

  client.start().catch(err => {
    window.showErrorMessage(`nanolang-lsp failed to start: ${err.message}`);
  });
}

function deactivate() {
  if (client) return client.stop();
}

module.exports = { activate, deactivate };
