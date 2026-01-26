/* eval_math.c - Math built-in functions for interpreter
 * Extracted from eval.c for better organization
 */

#define _POSIX_C_SOURCE 200809L

#include "eval_math.h"
#include "../nanolang.h"
#include <stdlib.h>
#include <math.h>

/* ==========================================================================
 * Math and Utility Built-in Functions
 * ========================================================================== */

Value builtin_abs(Value *args) {
    if (args[0].type == VAL_INT) {
        long long val = args[0].as.int_val;
        return create_int(val < 0 ? -val : val);
    } else if (args[0].type == VAL_FLOAT) {
        double val = args[0].as.float_val;
        return create_float(val < 0 ? -val : val);
    }
    fprintf(stderr, "Error: abs requires int or float argument\n");
    return create_void();
}

Value builtin_min(Value *args) {
    if (args[0].type == VAL_INT && args[1].type == VAL_INT) {
        long long a = args[0].as.int_val;
        long long b = args[1].as.int_val;
        return create_int(a < b ? a : b);
    } else if (args[0].type == VAL_FLOAT && args[1].type == VAL_FLOAT) {
        double a = args[0].as.float_val;
        double b = args[1].as.float_val;
        return create_float(a < b ? a : b);
    }
    fprintf(stderr, "Error: min requires two arguments of same type (int or float)\n");
    return create_void();
}

Value builtin_max(Value *args) {
    if (args[0].type == VAL_INT && args[1].type == VAL_INT) {
        long long a = args[0].as.int_val;
        long long b = args[1].as.int_val;
        return create_int(a > b ? a : b);
    } else if (args[0].type == VAL_FLOAT && args[1].type == VAL_FLOAT) {
        double a = args[0].as.float_val;
        double b = args[1].as.float_val;
        return create_float(a > b ? a : b);
    }
    fprintf(stderr, "Error: max requires two arguments of same type (int or float)\n");
    return create_void();
}

/* Advanced Math Functions */
Value builtin_sqrt(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(sqrt(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(sqrt((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: sqrt requires numeric argument\n");
    return create_void();
}

Value builtin_pow(Value *args) {
    double base, exponent;
    if (args[0].type == VAL_FLOAT) {
        base = args[0].as.float_val;
    } else if (args[0].type == VAL_INT) {
        base = (double)args[0].as.int_val;
    } else {
        fprintf(stderr, "Error: pow requires numeric arguments\n");
        return create_void();
    }

    if (args[1].type == VAL_FLOAT) {
        exponent = args[1].as.float_val;
    } else if (args[1].type == VAL_INT) {
        exponent = (double)args[1].as.int_val;
    } else {
        fprintf(stderr, "Error: pow requires numeric arguments\n");
        return create_void();
    }

    return create_float(pow(base, exponent));
}

Value builtin_floor(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(floor(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: floor requires numeric argument\n");
    return create_void();
}

Value builtin_ceil(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(ceil(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: ceil requires numeric argument\n");
    return create_void();
}

Value builtin_round(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(round(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: round requires numeric argument\n");
    return create_void();
}

/* Trigonometric Functions */
Value builtin_sin(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(sin(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(sin((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: sin requires numeric argument\n");
    return create_void();
}

Value builtin_cos(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(cos(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(cos((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: cos requires numeric argument\n");
    return create_void();
}

Value builtin_tan(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(tan(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(tan((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: tan requires numeric argument\n");
    return create_void();
}

Value builtin_atan2(Value *args) {
    double y = 0.0, x = 0.0;

    /* Get y value */
    if (args[0].type == VAL_FLOAT) {
        y = args[0].as.float_val;
    } else if (args[0].type == VAL_INT) {
        y = (double)args[0].as.int_val;
    } else {
        fprintf(stderr, "Error: atan2 requires numeric arguments\n");
        return create_void();
    }

    /* Get x value */
    if (args[1].type == VAL_FLOAT) {
        x = args[1].as.float_val;
    } else if (args[1].type == VAL_INT) {
        x = (double)args[1].as.int_val;
    } else {
        fprintf(stderr, "Error: atan2 requires numeric arguments\n");
        return create_void();
    }

    return create_float(atan2(y, x));
}
