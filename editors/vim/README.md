# Nanolang Vim Syntax Highlighting

Syntax highlighting for Nanolang in Vim and Neovim.

## Installation

### Manual Installation

1. Copy syntax file:
```bash
mkdir -p ~/.vim/syntax
cp editors/vim/nanolang.vim ~/.vim/syntax/
```

2. Copy filetype detection:
```bash
mkdir -p ~/.vim/ftdetect
cp editors/vim/ftdetect/nanolang.vim ~/.vim/ftdetect/
```

3. Open any `.nano` file

### Using Vim-Plug

Add to your `.vimrc` or `init.vim`:

```vim
Plug 'nanolang/nanolang.vim'  " (when published)
```

Or manually link this repository.

### Using Pathogen

```bash
cd ~/.vim/bundle
git clone https://github.com/nanolang/nanolang path/to/repo/editors/vim nanolang
```

## Features

- Keyword highlighting (fn, let, if, while, etc.)
- Type highlighting (int, float, string, custom types)
- Number and string literals
- Comments (line and block)
- Built-in function highlighting
- Operator highlighting
- S-expression folding support

## Configuration

Add to your `.vimrc` for enhanced Nanolang editing:

```vim
" Enable syntax highlighting
syntax on

" Set tab width for Nanolang files
autocmd FileType nanolang setlocal tabstop=4 shiftwidth=4 expandtab

" Enable folding for S-expressions
autocmd FileType nanolang setlocal foldmethod=syntax
```

## Testing

Open a `.nano` file and check:
```vim
:set filetype?
```

Should output: `filetype=nanolang`

