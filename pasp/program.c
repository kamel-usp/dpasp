#include <clingo.h>

#include <stdio.h>
#include <wchar.h>

#include "cutils.c"

typedef struct prob_fact {
  double p;
  const char *f;
  clingo_symbol_t cl_f;
} prob_fact_t;

typedef struct credal_fact {
  double l;
  double u;
  const char *f;
  clingo_symbol_t cl_f;
} credal_fact_t;

typedef struct query {
  clingo_symbol_t *Q;
  bool *Q_s;
  size_t Q_n;
  clingo_symbol_t *E;
  bool *E_s;
  size_t E_n;
} query_t;

void print_query(query_t *q) {
  size_t i;
  string_t s = {NULL, 0};
  bool has_E = q->E_n > 0;

  fputws(L"ℙ(", stdout);
  for (i = 0; i < q->Q_n; ++i) {
    if (!q->Q_s[i]) fputs("not ", stdout);
    if (!string_from_symbol(q->Q[i], &s)) goto cleanup;
    printf("%s", s.s);
    if (i != q->Q_n-1) fputs(", ", stdout);
    else {
      if (has_E) fputs(" | ", stdout);
      else fputs(")", stdout);
    }
  }
  for(i = 0; i < q->E_n; ++i) {
    if (!q->E_s[i]) fputs("not ", stdout);
    if (!string_from_symbol(q->E[i], &s)) goto cleanup;
    printf("%s", s.s);
    if (i != q->E_n-1) fputs(", ", stdout);
    else fputs(")", stdout);
  }

cleanup:
  free(s.s);
}

typedef struct program {
  const char *P;
  prob_fact_t *PF;
  size_t PF_n;
  query_t *Q;
  size_t Q_n;
  credal_fact_t *CF;
  size_t CF_n;
} program_t;


