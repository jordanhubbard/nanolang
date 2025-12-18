# Nanolang Emacs Mode

Major mode for editing Nanolang source files in Emacs.

## Installation

### Manual Installation

1. Copy the mode file to your Emacs load path:
```bash
mkdir -p ~/.emacs.d/lisp
cp editors/emacs/nanolang-mode.el ~/.emacs.d/lisp/
```

2. Add to your `~/.emacs` or `~/.emacs.d/init.el`:
```elisp
(add-to-list 'load-path "~/.emacs.d/lisp")
(require 'nanolang-mode)
```

3. Restart Emacs or evaluate the configuration:
```
M-x eval-buffer
```

### Using use-package

```elisp
(use-package nanolang-mode
  :load-path "path/to/repo/editors/emacs"
  :mode "\\.nano\\'")
```

### Using straight.el

```elisp
(straight-use-package
 '(nanolang-mode :type git :host github :repo "nanolang/nanolang"
                 :files ("editors/emacs/nanolang-mode.el")))
```

## Features

- **Syntax Highlighting**: Keywords, types, functions, operators, constants
- **Indentation**: Automatic indentation based on parentheses depth
- **Comments**: Line (`//`) and block (`/* */`) comment support
- **Auto Mode**: Automatically enables for `.nano` files
- **S-expression Navigation**: Standard Emacs paren-matching

## Usage

Open any `.nano` file and the mode will activate automatically.

### Key Bindings

- `RET` - New line with automatic indentation
- `C-M-f` / `C-M-b` - Move forward/backward by S-expression
- `C-M-k` - Kill S-expression
- `C-M-SPC` - Mark S-expression
- `M-;` - Comment/uncomment region

### Configuration

Add to your init file for enhanced Nanolang editing:

```elisp
(add-hook 'nanolang-mode-hook
  (lambda ()
    ;; Customize indentation
    (setq-local tab-width 4)
    (setq-local indent-tabs-mode nil)
    
    ;; Enable line numbers
    (display-line-numbers-mode 1)
    
    ;; Enable rainbow delimiters if installed
    (when (fboundp 'rainbow-delimiters-mode)
      (rainbow-delimiters-mode 1))
    
    ;; Enable show-paren-mode
    (show-paren-mode 1)))
```

## Customization

### Change Indentation Width

```elisp
(setq nanolang-indent-offset 2)
```

### Custom Font Faces

```elisp
(custom-set-faces
 '(nanolang-keyword-face ((t (:foreground "blue" :weight bold))))
 '(nanolang-type-face ((t (:foreground "green")))))
```

## Troubleshooting

**Mode not activating:**
```elisp
M-x nanolang-mode
```

**Check file association:**
```elisp
M-: (assoc "\\.nano\\'" auto-mode-alist)
```

**Reload mode:**
```elisp
M-x load-file RET ~/.emacs.d/lisp/nanolang-mode.el RET
```

