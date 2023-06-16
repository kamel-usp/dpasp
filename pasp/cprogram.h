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

ARRAY_DECL(prob_fact_t)
typedef array_prob_fact_t_t array_prob_fact_t;

typedef struct {
  double p;
  const char *f;
  PyObject *f_obj;
  bool is_prop;
  bool learnable;
  bool sharing;
  const char *unify;
  array_uint8_t_t PF;
  PyObject *unify_obj;
  PyObject *self;
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

typedef struct {
  /* Output probabilities. */
  float *P;
  /* Heads. */
  clingo_symbol_t *H;
  /* Bodies. */
  clingo_symbol_t *B;
  /* Sign of literals in bodies. */
  bool *S;
  /* Number of ground rules. */
  size_t n;
  /* Number of subgoals in the rule's body. */
  size_t k;
  /* Number of outcomes in the neural network. */
  size_t o;
  /* Derivative tensor data. */
  float *dw;
  bool learnable;
  PyObject *self;
} neural_rule_t;

typedef struct {
  /* Output probabilities. */
  float *P;
  /* Heads. */
  clingo_symbol_t *H;
  /* Bodies. */
  clingo_symbol_t *B;
  /* Sign of literals in bodies. */
  bool *S;
  /* Number of ground rules. */
  size_t n;
  /* Number of values. */
  size_t v;
  /* Number of subgoals in the rule's body. */
  size_t k;
  /* Number of outcomes in the neural network. */
  size_t o;
  /* Derivative tensor data. */
  float *dw;
  bool learnable;
  PyObject *self;
} neural_annot_disj_t;

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

typedef struct {
  /* Grounding rule. */
  const char *gr_rule;
  PyObject *py_gr_rule;
  /* Number of query variables. */
  size_t Q_n;
  /* Signs of query variables. */
  uint8_t *Q_s;
  /* Number of evidence variables. */
  size_t E_n;
  /* Signs of evidence variables. */
  uint8_t *E_s;
  /* VarQuery Python object. */
  PyObject *self;
} varquery_t;

typedef enum semantics {
  STABLE_SEMANTICS = 0,
  PARTIAL_SEMANTICS = 1,
  LSTABLE_SEMANTICS = 2,
  SMPROBLOG_SEMANTICS = 3,
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
  varquery_t *VQ;
  size_t VQ_n;
  credal_fact_t *CF;
  size_t CF_n;
  annot_disj_t *AD;
  size_t AD_n;

  const char *gr_P;
  PyObject *py_gr_P;
  bool is_ground;

  neural_rule_t *NR;
  size_t NR_n;
  neural_annot_disj_t *NA;
  size_t NA_n;

  /* Number of instances in test data. */
  size_t m_test;
  /* Number of instances in train data. */
  size_t m_train;
  /* Batch size. */
  size_t batch;

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

bool update_pr_neural_rule(neural_rule_t *nr);
bool update_forward_neural_rule(neural_rule_t *nr, size_t start, size_t end);
bool backward_neural_rule(neural_rule_t *nr);
void free_neural_rule_contents(neural_rule_t *nr);
void free_neural_rule(neural_rule_t *nr);

bool update_pr_neural_annot_disj(neural_annot_disj_t *na);
bool update_forward_neural_annot_disj(neural_annot_disj_t *na, size_t start, size_t end);
bool backward_neural_annot_disj(neural_annot_disj_t *na);
void free_neural_annot_disj_contents(neural_annot_disj_t *na);
void free_neural_annot_disj_rule(neural_annot_disj_t *na);

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
bool from_python_neural_rule(PyObject *py_nr, neural_rule_t *nr);
bool from_python_program(PyObject *py_P, program_t *P);

#endif
