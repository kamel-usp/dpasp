#include "cinf.h"
#include "cutils.h"

double prob_total_choice(prob_fact_t *phi, size_t n, size_t CF_n, total_choice_t *theta,
    uint8_t *theta_ad) {
  size_t i = 0, ad_n = theta->ad_n;
  double p = 1.0;
  bool t;
  for (; i < n; ++i) {
    t = bitvec_GET(&theta->pf, i + CF_n);
    p *= t*phi[i].p + (!t)*(1.0-phi[i].p);
  }
  for (i = 0; i < ad_n; ++i) p *= theta->ad[i].P[theta_ad[i]];
  return p;
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
  free_total_choice_contents(&s->theta);
}

bool setup_conds(bool **cond_1, bool **cond_2, bool **cond_3, bool **cond_4, size_t n) {
  *cond_1 = (bool*) malloc(n);
  if (!(*cond_1)) goto nomem;
  *cond_2 = (bool*) malloc(n);
  if (!(*cond_2)) goto nomem;
  *cond_3 = (bool*) malloc(n);
  if (!(*cond_3)) goto nomem;
  *cond_4 = (bool*) malloc(n);
  if (!(*cond_4)) goto nomem;
  return true;
nomem:
  free(*cond_1); free(*cond_2);
  free(*cond_3); free(*cond_4);
  *cond_1 = *cond_2 = *cond_3 = *cond_4 = NULL;
  goto nomem;
}

bool setup_counts(size_t **count_q_e, size_t **count_e, size_t **count_partial_q_e, size_t n) {
  *count_q_e = (size_t*) malloc(n);
  if (!(*count_q_e)) goto nomem;
  *count_e = (size_t*) malloc(n);
  if (!(*count_e)) goto nomem;
  if (count_partial_q_e) {
    *count_partial_q_e = (size_t*) malloc(n);
    if (!(*count_partial_q_e)) goto nomem;
  }
  return true;
nomem:
  free(*count_q_e); free(*count_e); free(*count_partial_q_e);
  *count_q_e = *count_e = *count_partial_q_e = NULL;
  return false;
}

bool setup_abcd(double **a, double **b, double **c, double **d, size_t n, size_t s) {
  *a = (double*) calloc(n, s);
  if (!(*a)) goto nomem;
  *b = (double*) calloc(n, s);
  if (!(*b)) goto nomem;
  if (c) {
    *c = (double*) calloc(n, s);
    if (!(*c)) goto nomem;
  } if (d) {
    *d = (double*) calloc(n, s);
    if (!(*d)) goto nomem;
  }
  return true;
nomem:
  free(*a); free(*b);
  free(*c); free(*d);
  *a = *b = *c = *d = NULL;
  return false;
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
bool _incr_total_choice_ad(uint8_t *theta, annot_disj_t *ad, size_t i, size_t ad_n) {
  if (i == ad_n-1) return (theta[i] = (theta[i] + 1) % ad[i].n) == 0;
  bool c = _incr_total_choice_ad(theta, ad, i+1, ad_n);
  bool l = theta[i] == ad[i].n-1;
  theta[i] = (theta[i] + c) % ad[i].n;
  return c && l;
}
/**
 * Recursive implementation of incrementing total_choice_t ADs.
 */
bool incr_total_choice_ad(total_choice_t *theta) {
  return !_incr_total_choice_ad(theta->theta_ad, theta->ad, 0, theta->ad_n);
}

void print_total_choice(total_choice_t *theta) {
  printf("Total choice:\nPF: ");
  bitvec_print(&theta->pf);
  for (size_t i = 0; i < theta->ad_n; ++i)
    printf("AD[%lu] = %u\n", i, theta->theta_ad[i]);
}

size_t estimate_nprocs(size_t total_choice_n) {
  return (total_choice_n > log2(NUM_PROCS)) ? NUM_PROCS : (1 << total_choice_n);
}

int retr_free_proc(bool *busy_procs, size_t num_procs, pthread_mutex_t *wakeup,
    pthread_cond_t *avail) {
  size_t i;
  int id = -1;
  /* The line below does not produce a problematic race condition since it will, at worst, skip
   * the i-th busy_procs and have to iterate NUM_PROCS all over again. */
  pthread_mutex_lock(wakeup);
  while (true) {
    for (i = 0, id = -1; i < num_procs; ++i) {
      if (!busy_procs[i]) { id = i; break; }
    }
    if (id != -1) break;
    pthread_cond_wait(avail, wakeup);
  }
  busy_procs[id] = true;
  pthread_mutex_unlock(wakeup);
  return id;
}

bool dispatch_job(total_choice_t *theta, pthread_mutex_t *wakeup, bool *busy_procs, storage_t *S,
    size_t num_procs, threadpool pool, pthread_cond_t *avail, void (*compute_func)(void*)) {
  int id = retr_free_proc(busy_procs, num_procs, wakeup, avail);
  copy_total_choice(theta, &S[id].theta);
  return !(S[id].fail || thpool_add_work(pool, compute_func, &S[id]));
}

#define PROCS_STR(x) #x ",compete"
#define PROCS_XSTR(x) PROCS_STR(x)
#define NUM_PROCS_CONFIG_STR PROCS_XSTR(NUM_PROCS)

bool prepare_control(clingo_control_t **C, program_t *P, total_choice_t *theta,
    uint8_t *theta_ad, const char *nmodels, bool parallelize_clingo, const char *append) {
  size_t i;
  clingo_backend_t *back = NULL;
  /* Create new clingo controller. */
  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, C)) return false;
  /* Config to enumerate all models. */
  if (!setup_config(*C, nmodels, false)) return false;
  /* Add the purely logical part. */
  if (!clingo_control_add(*C, "base", NULL, 0, P->P)) return false;
  if (append) if (!clingo_control_add(*C, "base", NULL, 0, append)) return false;
  /* Add grounded probabilistic rules. */
  if (P->gr_P[0]) if (!clingo_control_add(*C, "base", NULL, 0, P->gr_P)) return false;
  /* Get the control's backend. */
  if (!clingo_control_backend(*C, &back)) return false;
  /* Startup the backend. */
  if (!clingo_backend_begin(back)) return false;
  /* Add the credal facts according to the total rule. */
  for (i = 0; i < P->CF_n; ++i) {
    clingo_atom_t a;
    if (!CHOICE_IS_TRUE(theta, i)) continue;
    if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &a)) return false;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) return false;
  }
  /* Add the probabilistic facts according to the total rule. */
  for (i = 0; i < P->PF_n; ++i) {
    clingo_atom_t a;
    if (!CHOICE_IS_TRUE(theta, i + P->CF_n)) continue;
    if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &a)) return false;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) return false;
  }
  /* Add the annotated disjunction rules according to the total rule encoded by theta_ad. */
  for (i = 0; i < theta->ad_n; ++i) {
    clingo_atom_t a;
    if (!clingo_backend_add_atom(back, &theta->ad[i].cl_F[theta_ad[i]], &a)) return false;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) return false;
  }
  /* Cleanup backend. */
  if (!clingo_backend_end(back)) return false;
  /* Ground atoms. */
  if(!clingo_control_ground(*C, GROUND_DEFAULT_PARTS, 1, NULL, NULL)) return false;
  return true;
}

bool setup_config(clingo_control_t *C, const char *nmodels, bool parallelize_clingo) {
  clingo_configuration_t *cfg = NULL;
  clingo_id_t cfg_root, cfg_sub;

  /* Get the control's configuration. */
  if (!clingo_control_configuration(C, &cfg)) return false;
  /* Set to enumerate all stable models. */
  if (!clingo_configuration_root(cfg, &cfg_root)) return false;
  if (!clingo_configuration_map_at(cfg, cfg_root, "solve.models", &cfg_sub)) return false;
  if (!clingo_configuration_value_set(cfg, cfg_sub, nmodels)) return false;
  if (parallelize_clingo) {
    /* Set parallel_mode to "NUM_PROCS,compete", where NUM_PROCS is the #procs in this machine. */
    if (!clingo_configuration_map_at(cfg, cfg_root, "solve.parallel_mode", &cfg_sub)) return false;
    if (!clingo_configuration_value_set(cfg, cfg_sub, NUM_PROCS_CONFIG_STR)) return false;
  }

  return true;
}

bool has_total_model(program_t *P, total_choice_t *theta, uint8_t *theta_ad, bool *has) {
  clingo_control_t *C = NULL;
  clingo_solve_handle_t *handle;
  clingo_solve_result_bitset_t res;
  /* Prepare control according to the stable semantics. */
  if (!prepare_control(&C, P->stable, theta, theta_ad, "1", false, NULL)) goto cleanup;
  /* Solve and determine if there exists a (total) model. */
  if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle)) goto cleanup;
  if (!clingo_solve_handle_get(handle, &res)) goto cleanup;
  *has = (res & clingo_solve_result_satisfiable);
  /* Cleanup. */
  clingo_control_free(C);
  return true;
cleanup:
  clingo_control_free(C);
  return false;
}


