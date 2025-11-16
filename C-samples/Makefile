# Makefile for C sample programs

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
SDL_CFLAGS = $(shell pkg-config --cflags sdl2 2>/dev/null || echo "-I/opt/homebrew/include/SDL2 -I/usr/local/include/SDL2")
SDL_LIBS = $(shell pkg-config --libs sdl2 2>/dev/null || echo "-L/opt/homebrew/lib -L/usr/local/lib -lSDL2")

# Check if SDL2_ttf is available
SDL2_TTF_AVAILABLE := $(shell pkg-config --exists sdl2_ttf 2>/dev/null && echo "yes" || echo "no")
ifeq ($(SDL2_TTF_AVAILABLE),yes)
    SDL_CFLAGS += $(shell pkg-config --cflags sdl2_ttf)
    SDL_LIBS += $(shell pkg-config --libs sdl2_ttf)
    CFLAGS += -DHAVE_SDL_TTF
else
    # Try common locations
    ifneq ($(wildcard /opt/homebrew/include/SDL2/SDL_ttf.h),)
        SDL_CFLAGS += -I/opt/homebrew/include/SDL2 -DHAVE_SDL_TTF
        SDL_LIBS += -L/opt/homebrew/lib -lSDL2_ttf
    else ifneq ($(wildcard /usr/local/include/SDL2/SDL_ttf.h),)
        SDL_CFLAGS += -I/usr/local/include/SDL2 -DHAVE_SDL_TTF
        SDL_LIBS += -L/usr/local/lib -lSDL2_ttf
    endif
endif

# Default target
all: checkers

# Checkers game
checkers: checkers.c
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o checkers checkers.c $(SDL_LIBS)

# Clean
clean:
	rm -f checkers

.PHONY: all clean

