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
	rm -rf .test_output

test: $(TARGET)
	@./test.sh

.PHONY: all clean test
