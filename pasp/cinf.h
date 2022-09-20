#ifndef _PASP_CINF
#define _PASP_CINF

#include "cprogram.h"
#include "carray.h"

typedef enum psemantics {
  CREDAL_SEMANTICS = 0,
  PLOG_SEMANTICS   = 1,
} psemantics_t;

double prob_total_choice(prob_fact_t *phi, size_t n, array_double_t *gr_pr, unsigned long long int theta);
unsigned long long int sample_total_choice(prob_fact_t *phi, size_t n, array_double_t *gr_pr);

typedef struct storage {
  bool *cond_1, *cond_2, *cond_3, *cond_4;
  size_t *count_q_e, *count_e, *count_partial_q_e;
  double *a, *b, *c, *d;
  array_bool_t (*Pn)[4];
  array_double_t (*K)[4];
  program_t *P;
  unsigned long long int theta;
  bool fail, *busy_procs, lstable_sat;
  size_t pid;
  pthread_mutex_t *mu, *wakeup;
  pthread_cond_t *avail;
} storage_t;

bool init_storage(storage_t *s, program_t *P, array_bool_t (*Pn)[4],
    array_double_t (*K)[4], size_t id, bool *busy_procs, pthread_mutex_t *mu,
    pthread_mutex_t *wakeup, pthread_cond_t *avail, bool lstable_sat);
void free_storage_contents(storage_t *s);

bool setup_conds(bool **cond_1, bool **cond_2, bool **cond_3, bool **cond_4, size_t n);
bool setup_counts(size_t **count_q_e, size_t **count_e, size_t **count_partial_q_e, size_t n);
bool setup_abcd(double **a, double **b, double **c, double **d, size_t n, size_t s);

#endif

