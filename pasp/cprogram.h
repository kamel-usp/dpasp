#ifndef _PASP_CPROGRAM
#define _PASP_CPROGRAM

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <clingo.h>
#include "carray.h"

typedef struct {
  double p;
  const char *f;
  PyObject *f_obj;
  clingo_symbol_t cl_f;
  bool learnable;
  PyObject *self;
} prob_fact_t;

typedef struct {
  double p;
  const char *f;
  PyObject *f_obj;
  bool is_prop;
  const char *unify;
  PyObject *unify_obj;
} prob_rule_t;

typedef struct {
  double l;
  double u;
  const char *f;
  PyObject *f_obj;
  clingo_symbol_t cl_f;
} credal_fact_t;

typedef struct {
  double *P;
  const char **F;
  PyObject **F_obj;
  clingo_symbol_t *cl_F;
  size_t n;
  bool learnable;
  PyObject *self;
} annot_disj_t;

#define QUERY_TERM_NEG 0
#define QUERY_TERM_POS 1
#define QUERY_TERM_UND 2

typedef struct {
  clingo_symbol_t *Q;
  uint8_t *Q_s;
  size_t Q_n;
  clingo_symbol_t *Q_u; /* Potentially true auxiliary variables for partial semantics. */

  clingo_symbol_t *E;
  uint8_t *E_s;
  size_t E_n;
  clingo_symbol_t *E_u; /* Potentially true auxiliary variables for partial semantics. */
} query_t;

typedef enum semantics {
  STABLE_SEMANTICS = 0,
  PARTIAL_SEMANTICS = 1,
  LSTABLE_SEMANTICS = 2,
} semantics_t;

typedef struct program {
  const char *P;
  PyObject *P_obj;
  prob_fact_t *PF;
  size_t PF_n;
  prob_rule_t *PR;
  size_t PR_n;
  query_t *Q;
  size_t Q_n;
  credal_fact_t *CF;
  size_t CF_n;
  annot_disj_t *AD;
  size_t AD_n;

  const char *gr_P;
  PyObject *py_gr_P;

  semantics_t sem;
  struct program *stable;
  PyObject *py_P;
} program_t;

void print_prob_fact(prob_fact_t *pf);
void free_prob_fact_contents(prob_fact_t *pf);
void free_prob_fact(prob_fact_t *pf);

void print_prob_rule(prob_rule_t *pr);
void free_prob_rule_contents(prob_rule_t *pr);
void free_prob_rule(prob_rule_t *pr);

void print_credal_fact(credal_fact_t *cf);
void free_credal_fact_contents(credal_fact_t *cf);
void free_credal_fact(credal_fact_t *cf);

void print_annot_disj(annot_disj_t *ad);
void free_annot_disj_contents(annot_disj_t *ad);
void free_annot_disj(annot_disj_t *ad);

bool print_query(query_t *Q);
void free_query_contents(query_t *Q);
void free_query(query_t *Q);

void print_program(program_t *P);
void free_program_contents(program_t *P);
void free_program(program_t *P);

bool from_python_prob_rule(PyObject *py_pr, prob_rule_t *pr);
bool from_python_prob_fact(PyObject *py_pf, prob_fact_t *pf);
bool from_python_credal_fact(PyObject *py_cf, credal_fact_t *cf);
bool from_python_query(PyObject *py_q, query_t *q, semantics_t sem);
bool from_python_ad(PyObject *py_ad, annot_disj_t *ad);
bool from_python_program(PyObject *py_P, program_t *P);

#endif
