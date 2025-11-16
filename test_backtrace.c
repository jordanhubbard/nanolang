#include "src/nanolang.h"

void test_function(void) {
    assert(0 && "This should trigger backtrace");
}

int main(void) {
    test_function();
    return 0;
}
