CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
TARGET = nano
SOURCES = main.c lexer.c parser.c eval.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.c nanolang.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

test: $(TARGET)
	@echo "Running basic tests..."
	@echo "5 + 3" | ./$(TARGET) | grep -q "8" && echo "✓ Addition test passed" || echo "✗ Addition test failed"
	@echo "10 - 4" | ./$(TARGET) | grep -q "6" && echo "✓ Subtraction test passed" || echo "✗ Subtraction test failed"
	@echo "3 * 4" | ./$(TARGET) | grep -q "12" && echo "✓ Multiplication test passed" || echo "✗ Multiplication test failed"
	@echo "Testing variable assignment..."
	@./$(TARGET) examples/variables.nano && echo "✓ Variables test passed" || echo "✗ Variables test failed"
	@./$(TARGET) examples/fibonacci.nano && echo "✓ Fibonacci test passed" || echo "✗ Fibonacci test failed"

.PHONY: all clean test
