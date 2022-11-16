#ifndef _PASP_COPTIMIZE
#define _PASP_COPTIMIZE

#include <stdbool.h>
#include <stdlib.h>

double bfca(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, int maxmin, size_t tries, bool smp);

void bf(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, double *low, double *up, bool smp);

void bf_minmax(double *X, bool *S_a, bool *S_b, bool* S_c, bool* S_d, double *C_a,
    double *C_b, double *C_c, double *C_d, double *L, double *U, size_t n_a, size_t n_b,
    size_t n_c, size_t n_d, size_t m, double *low, double *up);

#endif
