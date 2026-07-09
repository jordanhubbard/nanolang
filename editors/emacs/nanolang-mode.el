;;; nanolang-mode.el --- Major mode for Nanolang -*- lexical-binding: t; -*-

;; Copyright (C) 2024 Nanolang Team

;; Author: Nanolang Team
;; Keywords: languages
;; Version: 1.0.0
;; Package-Requires: ((emacs "24.3"))

;;; Commentary:

;; Major mode for editing Nanolang source files.
;; Provides syntax highlighting, indentation, and comment support.

;;; Code:

(defvar nanolang-mode-syntax-table
  (let ((table (make-syntax-table)))
    ;; Comments
    (modify-syntax-entry ?/ ". 124" table)
    (modify-syntax-entry ?* ". 23b" table)
    (modify-syntax-entry ?\n ">" table)
    
    ;; Strings
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\\ "\\" table)
    
    ;; Parentheses
    (modify-syntax-entry ?\( "()" table)
    (modify-syntax-entry ?\) ")(" table)
    (modify-syntax-entry ?\[ "(]" table)
    (modify-syntax-entry ?\] ")[" table)
    (modify-syntax-entry ?\{ "(}" table)
    (modify-syntax-entry ?\} "){" table)
    
    table)
  "Syntax table for `nanolang-mode'.")

(defvar nanolang-keywords
  '("fn" "let" "mut" "set" "return" "if" "else" "while" "for"
    "break" "continue" "match" "case" "default" "pub" "mod"
    "import" "as" "extern" "module" "struct" "union" "enum"
    "typedef" "const" "shadow" "assert" "opaque")
  "Nanolang keywords.")

(defvar nanolang-types
  '("int" "float" "bool" "string" "void" "array" "Result" "Option")
  "Nanolang primitive types.")

(defvar nanolang-builtin-functions
  '("println" "print" "len" "get" "set" "push" "pop"
    "map" "reduce" "filter" "fold" "cast" "sizeof")
  "Nanolang built-in functions.")

(defvar nanolang-constants
  '("true" "false" "null")
  "Nanolang constants.")

(defvar nanolang-font-lock-keywords
  `(
    ;; Keywords
    (,(regexp-opt nanolang-keywords 'words) . font-lock-keyword-face)
    
    ;; Types
    (,(regexp-opt nanolang-types 'words) . font-lock-type-face)
    ("\\<[A-Z][a-zA-Z0-9_]*\\>" . font-lock-type-face)
    
    ;; Built-in functions
    (,(regexp-opt nanolang-builtin-functions 'words) . font-lock-builtin-face)
    
    ;; Constants
    (,(regexp-opt nanolang-constants 'words) . font-lock-constant-face)
    
    ;; Numbers
    ("\\<[0-9]+\\.[0-9]+\\>" . font-lock-constant-face)
    ("\\<[0-9]+\\>" . font-lock-constant-face)
    ("\\<0x[0-9a-fA-F]+\\>" . font-lock-constant-face)
    
    ;; Function definitions
    ("\\<fn\\s-+\\([a-z_][a-zA-Z0-9_]*\\)" 1 font-lock-function-name-face)
    
    ;; Function calls
    ("\\<\\([a-z_][a-zA-Z0-9_]*\\)\\s-*(" 1 font-lock-function-name-face)
    
    ;; Operators
    ("[-+*/%<>=!&|]+" . font-lock-builtin-face))
  "Keyword highlighting for `nanolang-mode'.")

(defun nanolang-indent-line ()
  "Indent current line for Nanolang."
  (interactive)
  (let ((indent-level 0)
        (current-line (line-number-at-pos)))
    (save-excursion
      (beginning-of-line)
      ;; Count open parens/braces on previous lines
      (when (> current-line 1)
        (forward-line -1)
        (end-of-line)
        (let ((paren-depth (car (syntax-ppss))))
          (setq indent-level (* paren-depth 4)))))
    
    ;; Indent current line
    (indent-line-to indent-level)))

(defvar nanolang-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "RET") 'newline-and-indent)
    map)
  "Keymap for `nanolang-mode'.")

;;;###autoload
(define-derived-mode nanolang-mode prog-mode "Nanolang"
  "Major mode for editing Nanolang source files."
  :syntax-table nanolang-mode-syntax-table
  
  ;; Comments
  (setq-local comment-start "// ")
  (setq-local comment-end "")
  (setq-local comment-start-skip "//+\\s-*")
  
  ;; Font lock
  (setq-local font-lock-defaults '(nanolang-font-lock-keywords))
  
  ;; Indentation
  (setq-local indent-line-function 'nanolang-indent-line)
  (setq-local tab-width 4)
  (setq-local indent-tabs-mode nil))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.nano\\'" . nanolang-mode))

(provide 'nanolang-mode)

;;; nanolang-mode.el ends here

