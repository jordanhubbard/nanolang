# Checkers Game

A simple checkers game implemented in C using SDL2, designed to run as an X11 client.

## Requirements

- SDL2 development libraries (required)
- SDL2_ttf development libraries (optional - for nicer text rendering)
- C compiler (gcc or clang)
- X11 server (for running the game)

## Installation

### macOS (using Homebrew)

```bash
brew install sdl2
brew install sdl2_ttf  # Optional but recommended for better text rendering
make checkers
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install libsdl2-dev libsdl2-ttf-dev
make checkers
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install SDL2-devel SDL2_ttf-devel
make checkers
```

**Note:** If SDL2_ttf is not installed, the game will use a simple bitmap font renderer for the game over message. The game will still work, but the text will be simpler.

## Building

```bash
make checkers
```

This will create `bin/checkers`.

## Running

```bash
./bin/checkers
```

## How to Play

- **Red pieces** (top) are the player's pieces
- **Black pieces** (bottom) are the AI opponent's pieces
- Click on one of your pieces to select it (it will be highlighted in yellow)
- Click on a valid destination square to move
- Pieces move diagonally forward one square
- If you can jump over an opponent's piece, you must do so
- When a piece reaches the opposite end of the board, it becomes a king (indicated by a gold circle)
- Kings can move diagonally in any direction
- The game ends when one player has no pieces remaining

## Game Rules

- Regular pieces can only move forward diagonally
- Kings can move diagonally in any direction
- If a jump is available, it is mandatory
- Multiple jumps in a single turn are allowed
- Jumped pieces are removed from the board

## Troubleshooting

If you get an error about SDL2 not being found:

1. Make sure SDL2 is installed (`brew install sdl2` on macOS)
2. Check that the SDL2 headers are in a standard location
3. You may need to adjust the include paths in the Makefile

If the window doesn't appear:

- Make sure you're running in an X11 environment
- On macOS, you may need XQuartz installed for X11 support
- Alternatively, SDL2 should work natively on macOS without X11

