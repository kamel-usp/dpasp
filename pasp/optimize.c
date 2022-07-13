#include <stdio.h>
#include <stdlib.h>

#define bool char

double f(double *X, bool *S, double *C, int n, int m) {
  int i, j, u;
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

#define BFCA_MAXIMIZE -1
#define BFCA_MINIMIZE 1

/* Brute-force coordinate descent. */
double bfca(double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, double *L, double *U,
    int n_a, int n_b, int m, int maxmin, int tries) {
  double est, lest, best = 1;
  double a_l, a_u, b_l, b_u, l, u;
  int i, t, r;
  for (t = 0; t < tries; ++t) {
    r = rand() % (1 << m);
    for (i = 0; i < m; ++i) X[i] = (r >> i) % 2 ? L[i] : U[i];
    est = 0;
    lest = -1;
    while (est > lest) {
      for (i = 0; i < m; ++i) {
        X[i] = L[i];
        a_l = f(X, S_a, C_a, n_a, m);
        b_l = f(X, S_b, C_b, n_b, m);
        X[i] = U[i];
        a_u = f(X, S_a, C_a, n_a, m);
        b_u = f(X, S_b, C_b, n_b, m);
        l = maxmin*(a_l/(a_l+b_l));
        u = maxmin*(a_u/(a_u+b_u));
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

int main() {
  double C_a[] = {0.2, 0.4, 0.3, 0.5};
  double C_b[] = {0.7, 0.1, 0.2, 0.8, 0.3};
  bool S_a[] = {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0};
  bool S_b[] = {1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0};
  double X[] = {0, 0, 0};
  double L[] = {0.3, 0.5, 0.0};
  double U[] = {0.7, 0.5, 1.0};
  int n_a = 4, n_b = 5, m = 3, i;
  printf("Minimized value: %f\n", bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MINIMIZE, 3));
  for (i = 0; i < m; ++i) printf(" %f", X[i]);
  printf("\nMaximized value: %f\n", bfca(X, S_a, S_b, C_a, C_b, L, U, n_a, n_b, m, BFCA_MAXIMIZE, 3));
  for (i = 0; i < m; ++i) printf(" %f", X[i]);
  putchar('\n');
  return 0;
}
