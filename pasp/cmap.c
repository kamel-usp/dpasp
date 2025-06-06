#include "cmap.h"
#include <string.h>

bool init_map_mapping(map_mapping_t *M, query_t *Q) {
  if (Q->O_n >= 64) goto overflow;
  M->n = 1 << Q->O_n;
  M->Q = Q;
  M->P = (double*) calloc(M->n, sizeof(double));
  if (!array_uint64_t_init(&M->C)) goto memerr;
  return true;
memerr:
  PyErr_SetString(PyExc_MemoryError, "no memory!");
  return false;
overflow:
  PyErr_SetString(PyExc_MemoryError, "number of atoms in MAP query overflows: >= 64!");
  return false;
}

bool count_map_mapping(map_mapping_t *M, const clingo_model_t *model) {
  uint64_t b = 0;
  bool c;
  for (int i = 0; i < M->Q->O_n; ++i) {
    if (!clingo_model_contains(model, M->Q->O[i], &c)) return false;
    b |= (1 << i) & c;
  }
  return array_uint64_t_append(&M->C, c);
}

void accumulate_map_mapping(map_mapping_t *M, double u)  {
  for (int i = 0; i < M->C.n; ++i) M->P[M->C.d[i]] += u;
  array_uint64_t_clear(&M->C);
}

uint64_t argmax_map_mapping(map_mapping_t *M, double *p) {
  double m = 0;
  uint64_t i;
  for (i = 0; i < M->n; ++i)
    if (M->P[i] > m) m = M->P[i];
  *p = m;
  return i;
}

bool append_literal_str(clingo_symbol_t s, bool l, char *b, size_t *j) {
  size_t n;
  if (!clingo_symbol_to_string_size(s, &n)) return false;
  if (!clingo_symbol_to_string(s, b, n)) return false;
  *j = n;
  return true;
}
bool print_vals_map_mapping(map_mapping_t *M, uint64_t X) {
  char S[4096] = {0};
  size_t l = 0, j = 0;
  for (size_t i = 0; i < M->Q->O_n; ++i) {
    if (!append_literal_str(M->Q->O[i], X & (1 << i), S + l, &j)) return false;
    l += j;
    S[l++] = ','; S[l++] = ' ';
  }
  return true;
}

void reset_map_mapping(map_mapping_t *M) { memset(M->P, 0, M->n*sizeof(double)); }

void free_contents_map_mapping(map_mapping_t *M) {
  free(M->P);
  array_uint64_t_free_contents(&M->C);
}
