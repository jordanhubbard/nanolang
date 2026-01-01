/* Shadow test execution stub - interpreter removed
 * 
 * Shadow tests are now compiled into the final binary and execute
 * when the program runs. This stub allows compilation to proceed
 * without the interpreter's eval.c.
 */

#include "nanolang.h"
#include <stdio.h>

/* Stub: Shadow tests now execute in compiled binary, not during compilation */
bool run_shadow_tests(ASTNode *program, Environment *env) {
    (void)program;
    (void)env;
    
    /* Shadow tests will run when the compiled binary executes */
    /* No need to interpret them during compilation anymore */
    
    return true;  /* Always succeed - tests run at runtime */
}
