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
	@for file in examples/hello.nano examples/calculator.nano examples/factorial.nano \
		examples/01_operators.nano examples/02_strings.nano examples/03_floats.nano \
		examples/04_loops_working.nano examples/05_mutable.nano examples/06_logical.nano \
		examples/07_comparisons.nano examples/08_types.nano examples/09_math.nano \
		examples/primes.nano; do \
		echo "\nTesting $$file..."; \
		./$(TARGET) $$file -o test.out 2>&1 | head -20 && echo "✓ $$file passed" || echo "✗ $$file failed"; \
	done
	@rm -f test.out test.out.c

.PHONY: all clean test
