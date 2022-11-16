#include "coptimize.h"

/* The polynomial to evaluate, where X are the variables, S are the signs of each factor, C are the
 * coefficients, n are the number of terms and m are the number of variables. For example, the
 * following polynomial
 *
 *   f(x, y, z) = 0.2*(1-x)*(1-y)*z+0.4*(1-x)*y*(1-z)+0.3*x*(1-y)*(1-z)+0.5*x*y*(1-z)
 *
 * under the evaluation f(0.2, 0.5, 0.7) would be represented as
 *
 *   X = {0.2, 0.5, 0.7}
 *   S = {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0}
 *   C = {0.2, 0.4, 0.3, 0.5}
 *   n = 4
 *   m = 3
 *
 * Note that |X| = m, |S| = n*m, and |C| = n.
 */
double f(double *X, bool *S, double *C, size_t n, size_t m) {
  size_t i, j, u;
  double s, y;
  for (i = s = 0; i < n; ++i) {
    y = 1;
    for (j = 0; j < m; ++j) {
      u = m*i+j;
      y *= S[u] ? X[j] : 1-X[j];
    }
    s += C[i]*y;
  }
  return s;
}

#define OPTIMIZE_BFCA     0
#define OPTIMIZE_BF       1
#define OPTIMIZE_BFCA_SMP 2
#define OPTIMIZE_BF_SMP   3

#define OPTIMIZE_IS_BF(x)      (x) & 1
#define OPTIMIZE_IS_BFCA(x)    !(OPTIMIZE_IS_BF(x))
#define OPTIMIZE_IS_SMP(x)     (x) & 2
#define OPTIMIZE_IS_NOT_SMP(x) !(OPTIMIZE_IS_SMP(x))

#define BFCA_MAXIMIZE -1
#define BFCA_MINIMIZE 1

#define bfca_min(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, tries, smp) \
  bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MINIMIZE, tries, smp)
#define bfca_max(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, tries, smp) \
  bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MAXIMIZE, tries, smp)

/* Brute-force coordinate descent.
 *
 * Array X are the coordinates to optimize, S_i, C_i, n_i, m - where i ∈ {a, b} are the polynomials
 * that compose the objective function g(X)=a(X)/(a(X)+b(X)) to be optimized, L and U are the lower
 * and upper probabilities respectively of the credal facts. Integer maxmin ∈ {-1, 1} and defines
 * whether to minimize or maximize the objective function (prefer BFCA_MINIMIZE and BFCA_MAXIMIZE
 * instead). Parameter tries tells the algorithm how many initialization resets are to be tried
 * for finding possibly global optima, and smp determines (if true) that the function should
 * override the objective function with g(X)=a(X).
 */
double bfca(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, int maxmin, size_t tries, bool smp) {
  double est, lest, best = 1;
  double a_l, a_u, b_l, b_u, l, u;
  size_t i, t, r;
  for (t = 0; t < tries; ++t) {
    r = rand() % (1 << m);
    for (i = 0; i < m; ++i) X[i] = ((r >> i) % 2) ? L[i] : U[i];
    est = 0;
    lest = -1;
    while (est > lest) {
      for (i = 0; i < m; ++i) {
        if (smp) {
          X[i] = L[i];
          l = maxmin*f(X, S_a, C_a, n_a, m);
          X[i] = U[i];
          u = maxmin*f(X, S_a, C_a, n_a, m);
        } else {
          X[i] = L[i];
          a_l = f(X, S_a, C_a, n_a, m);
          b_l = f(X, S_b, C_b, n_b, m);
          X[i] = U[i];
          a_u = f(X, S_a, C_a, n_a, m);
          b_u = f(X, S_b, C_b, n_b, m);
          l = a_l+b_l;
          if (l != 0) l = maxmin*(a_l/l);
          u = a_u+b_u;
          if (u != 0) u = maxmin*(a_u/u);
        }
        lest = est;
        if (l < u) {
          X[i] = L[i];
          est = l;
        } else {
          X[i] = U[i];
          est = u;
        }
      }
    }
    if (best > est) best = est;
  }
  return maxmin*best;
}

/* Brute-force.
 *
 * Array X are the coordinates to optimize, S_i, C_i, n_i, m - where i ∈ {a, b} are the polynomials
 * that compose the objective function g(X)=a(X)/(a(X)+b(X)) to be optimized, L and U are the lower
 * and upper probabilities respectively of the credal facts. Parameter low and up are pointers to
 * where the function should store the minimized and maximized values. This function is constrained
 * over 1 ≤ m ≤ 30 (any call above 30 would end up taking too long anyway). Parameter smp
 * determines (if true) that the function should override the objective function with g(X)=a(X)
 */
void bf(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    size_t n_a, size_t n_b, size_t m, double *low, double *up, bool smp) {
  size_t j;
  unsigned long long int k, i;
  double a, b, y;

  *low = 1.0; *up = 0.0;
  k = 1 << m;
  for (i = 0; i < k; ++i) {
    for (j = 0; j < m; ++j) X[j] = ((i >> j) % 2) ? L[j] : U[j];
    a = f(X, S_a, C_a, n_a, m);
    b = f(X, S_b, C_b, n_b, m);
    if (smp) {
      if (*low > a) *low = a;
      if (*up < b) *up = b;
    } else {
      y = a+b;
      if (y != 0) y = a/y;
      if (*low > y) *low = y;
      if (*up < y) *up = y;
    }
  }
}
void bf_minmax(double *X, bool *S_a, bool *S_b, bool* S_c, bool* S_d, double *C_a,
    double *C_b, double *C_c, double *C_d, double *L, double *U, size_t n_a, size_t n_b,
    size_t n_c, size_t n_d, size_t m, double *low, double *up) {
  size_t j;
  unsigned long long int k, i;
  double a, b, c, d, y, z;

  *low = 1.0; *up = 0.0;
  k = 1 << m;
  for (i = 0; i < k; ++i) {
    for (j = 0; j < m; ++j) X[j] = ((i >> j) % 2) ? L[j] : U[j];
    a = f(X, S_a, C_a, n_a, m);
    b = f(X, S_b, C_b, n_b, m);
    c = f(X, S_c, C_c, n_c, m);
    d = f(X, S_d, C_d, n_d, m);
    y = a+d;
    if (y != 0) y = a/y;
    z = b+c;
    if (z != 0) z = b/z;
    if (*low > y) *low = y;
    if (*up < z) *up = z;
  }
}
