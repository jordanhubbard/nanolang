'use strict';

const path = require('path');
const { workspace, window, debug } = require('vscode');
const {
  LanguageClient,
  TransportKind
} = require('vscode-languageclient/node');

let client;

// ---------------------------------------------------------------------------
// DAP: DebugAdapterDescriptorFactory — launches nanolang-dap over stdio
// ---------------------------------------------------------------------------
class NanoLangDebugAdapterFactory {
  constructor(serverPath) {
    this.serverPath = serverPath;
  }

  createDebugAdapterDescriptor(_session) {
    const { DebugAdapterExecutable } = require('vscode');
    return new DebugAdapterExecutable(this.serverPath, []);
  }
}

function activate(context) {
  const config = workspace.getConfiguration('nanolang');

  // -------------------------------------------------------------------------
  // LSP
  // -------------------------------------------------------------------------
  const lspPath = config.get('serverPath') || 'nanolang-lsp';

  const serverOptions = {
    command: lspPath,
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

  // -------------------------------------------------------------------------
  // DAP debug adapter registration
  // -------------------------------------------------------------------------
  const dapPath = config.get('debugServerPath') || 'nanolang-dap';

  const factory = new NanoLangDebugAdapterFactory(dapPath);
  context.subscriptions.push(
    debug.registerDebugAdapterDescriptorFactory('nanolang', factory)
  );

  // Register a dynamic debug configuration provider so VS Code shows
  // "Debug NanoLang File" in the run menu for .nano files.
  context.subscriptions.push(
    debug.registerDebugConfigurationProvider('nanolang', {
      provideDebugConfigurations(_folder) {
        return [
          {
            type: 'nanolang',
            request: 'launch',
            name: 'Debug NanoLang File',
            program: '${file}'
          }
        ];
      },
      resolveDebugConfiguration(_folder, cfg) {
        if (!cfg.type && !cfg.request && !cfg.name) {
          const editor = window.activeTextEditor;
          if (editor && editor.document.languageId === 'nanolang') {
            cfg.type    = 'nanolang';
            cfg.request = 'launch';
            cfg.name    = 'Debug NanoLang File';
            cfg.program = editor.document.fileName;
          }
        }
        if (!cfg.program) {
          return window.showInformationMessage('Cannot find a NanoLang file to debug').then(() => undefined);
        }
        return cfg;
      }
    })
  );
}

function deactivate() {
  if (client) return client.stop();
}

module.exports = { activate, deactivate };
