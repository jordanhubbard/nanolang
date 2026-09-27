#include <stdint.h>
#include <stdbool.h>
#include <string.h>
static int64_t calls;
int64_t scalar_calls(void) { return calls; }
int64_t scalar_integer(int64_t value) { ++calls; return value - 7; }
int64_t scalar_enum(int64_t value) { ++calls; return value + 1; }
bool scalar_boolean(bool value) { ++calls; return !value; }
uint8_t scalar_byte(uint8_t value) { ++calls; return (uint8_t)(value + 1); }
int64_t scalar_text(const char *value) { ++calls; return (int64_t)strlen(value); }
double scalar_mixed(int64_t integer, double number, const char *text) {
    ++calls; return (double)integer + number + (double)strlen(text);
}
void scalar_void(void) { ++calls; }
