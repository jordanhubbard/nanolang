include Makefile.gnu
.PHONY: observer-build
observer-build:
	$(CC) $(CFLAGS) -c src/main.c -o obj/main.o
	$(CC) $(CFLAGS) -c src/eval.c -o obj/eval.o
	$(CC) $(CFLAGS) -o $(COMPILER_C) $(COMPILER_OBJECTS) $(LDFLAGS)
