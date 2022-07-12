#include <stdio.h>

#define bool char

double fcmp(double *X, bool *S, double *C, int n, int m) {
  int i, j, u;
  double s, y;
  for (i = s = 0; i < n; ++i) {
    y = 1;
    for (j = 0; j < m; ++j) {
      u = n*i+j;
      y *= S[u] ? X[u] : 1-X[u];
    }
    s += C[i]*y;
  }
  return s;
}

double f(double *X, double *C, int n) {
  int i;
  double s;
  for (i = s = 0; i < n; ++i) s += X[i]*C[i];
  return s+C[n];
}

#define BFCA_MAXIMIZE -1
#define BFCA_MINIMIZE 1

/* Brute-force coordinate descent. */
double bfca(double *X, double *C_a, double *C_b, double *L, double *U, int n, int maxmin) {
  double est = 0, lest = -1;
  double a_l, a_u, b_l, b_u, l, u;
  int i;
  for (i = 0; i < n; ++i) X[i] = L[i];
  while (est > lest) {
    for (i = 0; i < n; ++i) {
      X[i] = L[i];
      a_l = f(X, C_a, n);
      b_l = f(X, C_b, n);
      X[i] = U[i];
      a_u = f(X, C_a, n);
      b_u = f(X, C_b, n);
      l = maxmin*(a_l/(a_l+b_l));
      u = maxmin*(a_u/(a_u+b_u));
      printf("l: %f, u: %f, a_l: %f, b_l: %f, a_u: %f, b_u: %f\n", l, u, a_l, b_l, a_u, b_u);
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
  return maxmin*est;
}

int main() {
  double C_a[] = {0.2, 0.4, 0.3, 0.5};
  double C_b[] = {0.7, 0.1, 0.2, 0.8};
  double X[] = {0.3, 0.5, 0};
  double L[] = {0.3, 0.5, 0.0};
  double U[] = {0.7, 0.5, 1.0};
  int n = 3, i;
  printf("Minimized value: %f\n", bfca(X, C_a, C_b, L, U, n, BFCA_MINIMIZE));
  for (i = 0; i < n-1; ++i) printf(" %f", X[i]);
  printf("\nMaximized value: %f\n", bfca(X, C_a, C_b, L, U, n, BFCA_MAXIMIZE));
  bfca(X, C_a, C_b, L, U, n, 1);
  for (i = 0; i < n-1; ++i) printf(" %f", X[i]);
  putchar('\n');
  return 0;
}
