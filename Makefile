CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -Isrc
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
COMPILER = $(BIN_DIR)/nanoc
INTERPRETER = $(BIN_DIR)/nano
COMMON_SOURCES = $(SRC_DIR)/lexer.c $(SRC_DIR)/parser.c $(SRC_DIR)/typechecker.c $(SRC_DIR)/eval.c $(SRC_DIR)/transpiler.c $(SRC_DIR)/env.c
COMMON_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(COMMON_SOURCES))
COMPILER_OBJECTS = $(COMMON_OBJECTS) $(OBJ_DIR)/main.o
INTERPRETER_OBJECTS = $(COMMON_OBJECTS) $(OBJ_DIR)/interpreter_main.o

all: $(COMPILER) $(INTERPRETER)

$(COMPILER): $(COMPILER_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(COMPILER) $(COMPILER_OBJECTS)

$(INTERPRETER): $(INTERPRETER_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(INTERPRETER) $(INTERPRETER_OBJECTS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/nanolang.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -f $(OBJ_DIR)/*.o $(COMPILER) $(INTERPRETER) *.out *.out.c tests/*.out tests/*.out.c
	rm -rf .test_output

test: $(COMPILER) $(INTERPRETER)
	@./test.sh

.PHONY: all clean test
