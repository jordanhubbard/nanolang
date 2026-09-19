/* I observe ordinary GPU copies through an exact four-argument kernel. */
__kernel void identity_add(__global const long *source, __global long *destination,
                           long count, long bias) {
    size_t index = get_global_id(0);
    if (index < (size_t)count) destination[index] = source[index] + bias;
}
