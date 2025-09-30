CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
TARGET = nanoc
SOURCES = main.c lexer.c parser.c typechecker.c eval.c transpiler.c env.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.c nanolang.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) *.out *.out.c tests/*.out tests/*.out.c

test: $(TARGET)
	@echo "Testing nanolang compiler..."
	@echo "\nCompiling hello.nano..."
	@./$(TARGET) examples/hello.nano -o hello.out && echo "✓ hello.nano compiled" || echo "✗ hello.nano failed"
	@echo "\nCompiling calculator.nano..."
	@./$(TARGET) examples/calculator.nano -o calculator.out && echo "✓ calculator.nano compiled" || echo "✗ calculator.nano failed"
	@echo "\nCompiling factorial.nano..."
	@./$(TARGET) examples/factorial.nano -o factorial.out && echo "✓ factorial.nano compiled" || echo "✗ factorial.nano failed"

.PHONY: all clean test
