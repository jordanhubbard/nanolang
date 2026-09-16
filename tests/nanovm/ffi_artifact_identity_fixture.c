/* I share a scalar entry name with the array fixture to test handle isolation.
 * I do not duplicate its unrelated exported array ABI data globals. */
#include <stdint.h>
int64_t nano_artifact_answer(void) { return ARTIFACT_ANSWER; }
