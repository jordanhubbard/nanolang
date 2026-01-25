# Readline Module MVP

Interactive line editing with history support.

```nano
from "modules/readline/readline.nano" import rl_readline, rl_add_history

fn main() -> int {
    let input: string = (rl_readline "prompt> ")
    (rl_add_history input)
    (println input)
    return 0
}

shadow main { assert true }
```

## Features

- Line editing (Emacs-style by default)
- Command history (up/down arrows)
- History file persistence
- Works on macOS (libedit) and Linux (GNU readline)
