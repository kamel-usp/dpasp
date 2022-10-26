#include "cinf.h"

double prob_total_choice(prob_fact_t *phi, size_t n, array_double_t *gr_pr, size_t CF_n,
    total_choice_t *theta, uint8_t *theta_ad) {
  size_t i = 0, m = gr_pr->n, ad_n = theta->ad_n;
  double p = 1.0;
  bool t;
  for (; i < n; ++i) {
    t = bitvec_GET(&theta->pf, i + CF_n);
    p *= t*phi[i].p + (!t)*(1.0-phi[i].p);
  }
  for (i = 0; i < m; ++i) {
    t = bitvec_GET(&theta->pf, n + i + CF_n);
    p *= t*(gr_pr->d[i]) + (!t)*(1.0-(gr_pr->d[i]));
  }
  if (ad_n > 0) for (i = 0; i < ad_n; ++i) p *= theta->ad[i].P[theta_ad[i]];
  return p;
}

static inline bool sample_pf(double p) { return (((double) rand())/RAND_MAX) <= p; }

unsigned long long int sample_total_choice(prob_fact_t *phi, size_t n, array_double_t *gr_pr) {
  unsigned long long int theta = 0;
  size_t i, m = gr_pr->n;

  for (i = 0; i < n; ++i) theta |= sample_pf(phi[i].p) << i;
  for (i = 0; i < m; ++i) theta |= sample_pf(gr_pr->d[i]) << (n + i);

  return theta;
}

bool init_storage(storage_t *s, program_t *P, array_bool_t (*Pn)[4],
    array_double_t (*K)[4], size_t id, bool *busy_procs, pthread_mutex_t *mu,
    pthread_mutex_t *wakeup, pthread_cond_t *avail, bool lstable_sat, size_t total_choice_n,
    annot_disj_t *ad, size_t ad_n) {
  s->cond_1 = s->cond_2 = s->cond_3 = s->cond_4 = NULL;
  s->count_q_e = s->count_e = s->count_partial_q_e = NULL;
  s->a = s->b = s->c = s->d = NULL;
  s->Pn = Pn; s->K = K; s->P = P;
  s->mu = mu; s->wakeup = wakeup; s->avail = avail;
  if (!setup_conds(&s->cond_1, &s->cond_2, &s->cond_3, &s->cond_4, P->Q_n*sizeof(bool))) goto error;
  if (!setup_counts(&s->count_q_e, &s->count_e, &s->count_partial_q_e, P->Q_n*sizeof(size_t))) goto error;
  if (!P->CF_n) { if (!setup_abcd(&s->a, &s->b, &s->c, &s->d, P->Q_n, sizeof(double))) goto error; }
  s->busy_procs = busy_procs; s->lstable_sat = lstable_sat;
  s->pid = id;
  s->fail = false;
  if (!init_total_choice(&s->theta, total_choice_n, ad, ad_n)) goto error;
  return true;
error:
  PyErr_SetString(PyExc_MemoryError, "could not allocate enough memory for init_storage!");
  return false;
}

void free_storage_contents(storage_t *s) {
  free(s->cond_1); free(s->cond_2); free(s->cond_3); free(s->cond_4);
  free(s->count_q_e); free(s->count_e); free(s->count_partial_q_e);
  if (!s->P->CF_n) { free(s->a); free(s->b); free(s->c); free(s->d); }
}

bool setup_conds(bool **cond_1, bool **cond_2, bool **cond_3, bool **cond_4, size_t n) {
  *cond_1 = (bool*) malloc(n);
  if (!(*cond_1)) return false;
  *cond_2 = (bool*) malloc(n);
  if (!(*cond_2)) return false;
  *cond_3 = (bool*) malloc(n);
  if (!(*cond_3)) return false;
  *cond_4 = (bool*) malloc(n);
  if (!(*cond_4)) return false;
  return true;
}

bool setup_counts(size_t **count_q_e, size_t **count_e, size_t **count_partial_q_e, size_t n) {
  *count_q_e = (size_t*) malloc(n);
  if (!(*count_q_e)) return false;
  *count_e = (size_t*) malloc(n);
  if (!(*count_e)) return false;
  *count_partial_q_e = (size_t*) malloc(n);
  if (!(*count_partial_q_e)) return false;
  return true;
}

bool setup_abcd(double **a, double **b, double **c, double **d, size_t n, size_t s) {
  *a = (double*) calloc(n, s);
  if (!(*a)) return false;
  *b = (double*) calloc(n, s);
  if (!(*b)) return false;
  *c = (double*) calloc(n, s);
  if (!(*c)) return false;
  *d = (double*) calloc(n, s);
  if (!(*d)) return false;
  return true;
}

bool init_total_choice(total_choice_t *theta, size_t n, annot_disj_t *ad, size_t m) {
  if (!bitvec_init(&theta->pf, n)) return false;
  bitvec_zeron(&theta->pf, n);
  theta->ad = ad;
  theta->ad_n = m;
  theta->theta_ad = m > 0 ? (uint8_t*) calloc(m, sizeof(uint8_t)) : NULL;
  return true;
}
void free_total_choice_contents(total_choice_t *theta) {
  bitvec_free_contents(&theta->pf);
  free(theta->theta_ad);
}

total_choice_t* copy_total_choice(total_choice_t *src, total_choice_t *dst) {
  if (!dst) {
    dst = (total_choice_t*) malloc(sizeof(total_choice_t));
    if (!init_total_choice(dst, src->pf.n, src->ad, src->ad_n)) return NULL;
  } else { dst->ad = src->ad; dst->ad_n = src->ad_n; }
  bitvec_copy(&src->pf, &dst->pf);
  if (src->ad_n > 0) memcpy(dst->theta_ad, src->theta_ad, src->ad_n*sizeof(uint8_t));
  return dst;
}

bool incr_total_choice(total_choice_t *theta) { return bitvec_incr(&theta->pf); }
