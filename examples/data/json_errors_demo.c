/* json_errors_demo.c - Demonstrate JSON diagnostic output */

#include <stdio.h>
#include "../src/json_diagnostics.h"

int main() {
    printf("JSON Diagnostics Demo\n");
    printf("=====================\n\n");
    
    /* Initialize */
    json_diagnostics_init();
    json_diagnostics_enable();
    
    /* Add some example errors */
    json_error("E0001", "Type mismatch in let statement",
               "test.nano", 15, 20,
               "Check the type annotation matches the assigned value");
    
    json_error("E0002", "Undefined function 'foo'",
               "test.nano", 42, 10,
               "Import the function or check for typos");
    
    json_warning("W0001", "Unused variable 'x'",
                 "test.nano", 8, 5,
                 "Remove the variable or use it in your code");
    
    json_warning("W0002", "Function 'bar' is missing a shadow test",
                 "test.nano", 30, 1,
                 "Add a shadow block to test this function");
    
    /* Output as JSON */
    printf("JSON Output:\n");
    printf("------------\n");
    json_diagnostics_output();
    
    /* Cleanup */
    json_diagnostics_cleanup();
    
    return 0;
}
