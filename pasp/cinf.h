#ifndef _PASP_CINF
#define _PASP_CINF

#include "cprogram.h"
#include "carray.h"
#include "../bitvector/bitvector.h"

#include <pthread.h>
#include "../thpool/thpool.h"

/* Total choice size. */
#define TOTAL_CHOICE_N(P) (P)->PF_n+(P)->gr_pr.n
#define TOTAL_CHOICE_NCREDAL(P) (P)->PF_n+(P)->gr_pr.n+(P)->CF_n

/* Get dimension (number of classes) of the i-th learnable AD in storage S. */
#define STORAGE_AD_DIM(P, S, i) ((P)->AD[(S)->I_A[(i)]].n)

typedef enum psemantics {
  CREDAL_SEMANTICS = 0,
  MAXENT_SEMANTICS   = 1,
} psemantics_t;

typedef struct _total_choice {
  bitvec_t pf;
  annot_disj_t *ad;
  size_t ad_n;
  uint8_t *theta_ad;
} total_choice_t;

bool init_total_choice(total_choice_t *theta, size_t n, annot_disj_t *ad, size_t m);
void free_total_choice_contents(total_choice_t *theta);
total_choice_t* copy_total_choice(total_choice_t *src, total_choice_t *dst);
bool incr_total_choice(total_choice_t *theta);
void print_total_choice(total_choice_t *theta);

double prob_total_choice(prob_fact_t *phi, size_t n, array_double_t *gr_pr, size_t CF_n,
    total_choice_t *theta, uint8_t *ad_i);

typedef struct storage {
  bool *cond_1, *cond_2, *cond_3, *cond_4;
  size_t *count_q_e, *count_e, *count_partial_q_e;
  double *a, *b, *c, *d;
  array_bool_t (*Pn)[4];
  array_double_t (*K)[4];
  program_t *P;
  total_choice_t theta;
  bool fail, *busy_procs, lstable_sat;
  size_t pid;
  pthread_mutex_t *mu, *wakeup;
  pthread_cond_t *avail;
} storage_t;

bool init_storage(storage_t *s, program_t *P, array_bool_t (*Pn)[4],
    array_double_t (*K)[4], size_t id, bool *busy_procs, pthread_mutex_t *mu,
    pthread_mutex_t *wakeup, pthread_cond_t *avail, bool lstable_sat, size_t total_choice_n,
    annot_disj_t *ad, size_t ad_n);
void free_storage_contents(storage_t *s);

bool setup_conds(bool **cond_1, bool **cond_2, bool **cond_3, bool **cond_4, size_t n);
bool setup_counts(size_t **count_q_e, size_t **count_e, size_t **count_partial_q_e, size_t n);
bool setup_abcd(double **a, double **b, double **c, double **d, size_t n, size_t s);

#ifndef NUM_PROCS
#define NUM_PROCS 1
#endif

size_t estimate_nprocs(size_t total_choice_n);

int retr_free_proc(bool *busy_procs, size_t num_procs, pthread_mutex_t *wakeup,
    pthread_cond_t *avail);
bool dispatch_job(total_choice_t *theta, pthread_mutex_t *wakeup,
    bool *busy_procs, storage_t *S, size_t num_procs, threadpool pool, pthread_cond_t *avail,
    void (*compute_func)(void*), bool has_ad, size_t j, size_t c);

/* Determine if the i-th position in PF total choice t is true. */
#define CHOICE_IS_TRUE(t, i) bitvec_GET(&(t)->pf, i)

bool prepare_control(clingo_control_t **C, program_t *P, total_choice_t *theta,
    uint8_t *theta_ad, const char *nmodels, bool parallelize_clingo, const char *append);

bool setup_config(clingo_control_t *C, const char *nmodels, bool parallelize_clingo);
bool has_total_model(program_t *P, total_choice_t *theta, uint8_t *theta_ad, bool *has);

#endif
