#include "cinf.h"
#include "cutils.h"
#include <string.h>

double prob_total_choice_prob(program_t *P, total_choice_t *theta) {
  prob_fact_t *PF = P->PF;
  size_t PF_n = P->PF_n, AD_n = P->AD_n, CF_n = P->CF_n;
  size_t i = 0;
  double p = 1.0;
  bool t;
  for (; i < PF_n; ++i) {
    t = bitvec_GET(&theta->pf, i + CF_n);
    p *= t*PF[i].p + (!t)*(1.0-PF[i].p);
  }
  for (i = 0; i < AD_n; ++i) p *= P->AD[i].P[theta->theta_ad[i]];
  return p;
}
double prob_total_choice_neural(program_t *P, total_choice_t *theta, size_t offset, bool train) {
  double p = 1.0;
  size_t r = P->CF_n + P->PF_n;
  size_t m = train*P->m_train + (!train)*P->m_test;
  for (size_t i = 0; i < P->NR_n; ++i) {
    float *prob = P->NR[i].P + offset*P->NR[i].o;
    for (size_t j = 0; j < P->NR[i].n; ++j)
      for (size_t o = 0; o < P->NR[i].o; ++o) {
        bool t = bitvec_GET(&theta->pf, r++);
        double q = prob[j*P->NR[i].o*m+o];
        p *= t*q + (!t)*(1.0-q);
      }
  }
  r = P->AD_n;
  for (size_t i = 0; i < P->NA_n; ++i) {
    float *prob = P->NA[i].P + offset*P->NA[i].v*P->NA[i].o;
    for (size_t j = 0; j < P->NA[i].n; ++j)
      for (size_t o = 0; o < P->NA[i].o; ++o)
        p *= prob[j*m*P->NA[i].v*P->NA[i].o + o*P->NA[i].v + theta->theta_ad[r++]];
  }
  return p;
}
double prob_total_choice(program_t *P, total_choice_t *theta) {
  return prob_total_choice_prob(P, theta)*prob_total_choice_neural(P, theta, 0, false);
}
double prob_total_choice_ground(array_prob_fact_t *PF, total_choice_t *theta) {
  double p = 1.0;
  bool t;
  for (size_t i = 0; i < PF->n; ++i) {
    t = bitvec_GET(&theta->pf, i);
    p *= t*PF->d[i].p + (!t)*(1.0-PF->d[i].p);
  }
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
  s->fail = s->warn = false;
  if (!init_total_choice(&s->theta, total_choice_n, P)) goto error;
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

bool _init_total_choice(total_choice_t *theta, size_t n, size_t m) {
  if (!bitvec_init(&theta->pf, n)) return false;
  bitvec_zeron(&theta->pf, n);
  theta->ad_n = m;
  theta->theta_ad = (uint8_t*) calloc(m, sizeof(uint8_t));
  return true;
}
bool init_total_choice(total_choice_t *theta, size_t n, program_t *P) {
  size_t l = P->AD_n;
  for (size_t i = 0; i < P->NA_n; ++i) l += P->NA[i].n*P->NA[i].o;
  return _init_total_choice(theta, n, l);
}
void free_total_choice_contents(total_choice_t *theta) {
  bitvec_free_contents(&theta->pf);
  free(theta->theta_ad);
}

size_t get_num_facts(program_t *P) {
  size_t n = P->PF_n + P->CF_n;
  for (size_t i = 0; i < P->NR_n; ++i) n += P->NR[i].n*P->NR[i].o;
  return n;
}

total_choice_t* copy_total_choice(total_choice_t *src, total_choice_t *dst) {
  if (!dst) {
    dst = (total_choice_t*) malloc(sizeof(total_choice_t));
    if (!_init_total_choice(dst, src->pf.n, src->ad_n)) return NULL;
  } else dst->ad_n = src->ad_n;
  bitvec_copy(&src->pf, &dst->pf);
  if (src->ad_n > 0) memcpy(dst->theta_ad, src->theta_ad, src->ad_n*sizeof(uint8_t));
  return dst;
}

bool incr_total_choice(total_choice_t *theta) {
  return !theta->pf.n ? false : bitvec_incr(&theta->pf);
}
bool _incr_total_choice_ad(uint8_t *theta, annot_disj_t *ad, size_t i, size_t ad_n) {
  if (!ad_n) return true;
  if (i == ad_n-1) return (theta[i] = (theta[i] + 1) % ad[i].n) == 0;
  bool c = _incr_total_choice_ad(theta, ad, i+1, ad_n);
  bool l = theta[i] == ad[i].n-1;
  theta[i] = (theta[i] + c) % ad[i].n;
  return c && l;
}
bool _incr_total_choice_nad(uint8_t *theta, neural_annot_disj_t *nad, size_t i, size_t j, size_t a,
    size_t nad_n) {
  if (!nad_n) return true;
  if (a == nad_n-1) return (theta[a] = (theta[a] + 1) % nad[i].v) == 0;
  if (j == nad[i].n*nad[i].o) j = 0, ++i;
  bool c = _incr_total_choice_nad(theta, nad, i, j+1, a+1, nad_n);
  bool l = theta[a] == nad[i].v-1;
  theta[a] = (theta[a] + c) % nad[i].v;
  return c && l;
}
/**
 * Recursive implementation of incrementing total_choice_t ADs.
 */
bool incr_total_choice_ad(total_choice_t *theta, program_t *P) {
  return !(_incr_total_choice_ad(theta->theta_ad, P->AD, 0, P->AD_n) &&
    _incr_total_choice_nad(theta->theta_ad + P->AD_n, P->NA, 0, 0, 0, theta->ad_n - P->AD_n));
}

void print_total_choice(total_choice_t *theta) {
  wprintf(L"Total choice:\nPF: ");
  bitvec_wprint(&theta->pf);
  for (size_t i = 0; i < theta->ad_n; ++i)
    wprintf(L"AD[%lu] = %u\n", i, theta->theta_ad[i]);
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

bool dispatch_job_with_payload(total_choice_t *theta, pthread_mutex_t *wakeup, bool *busy_procs,
    storage_t *S, size_t num_procs, threadpool pool, pthread_cond_t *avail, int id,
    void (*compute_func)(void*), void *payload) {
  copy_total_choice(theta, &S[id].theta);
  return !(S[id].fail || thpool_add_work(pool, compute_func, payload));
}
bool dispatch_job(total_choice_t *theta, pthread_mutex_t *wakeup, bool *busy_procs, storage_t *S,
    size_t num_procs, threadpool pool, pthread_cond_t *avail, void (*compute_func)(void*)) {
  int id = retr_free_proc(busy_procs, num_procs, wakeup, avail);
  return dispatch_job_with_payload(theta, wakeup, busy_procs, S, num_procs, pool, avail, id,
      compute_func, (void*) &S[id]);
}

#define PROCS_STR(x) #x ",compete"
#define PROCS_XSTR(x) PROCS_STR(x)
#define NUM_PROCS_CONFIG_STR PROCS_XSTR(NUM_PROCS)

bool add_facts_from_total_choice(clingo_control_t *C, array_prob_fact_t *PF, total_choice_t *theta) {
  clingo_backend_t *back;
  if (!clingo_control_backend(C, &back)) return false;
  if (!clingo_backend_begin(back)) goto cleanup;
  for (size_t i = 0; i < PF->n; ++i) {
    clingo_atom_t a;
    if (!CHOICE_IS_TRUE(theta, i)) continue;
    if (!clingo_backend_add_atom(back, &PF->d[i].cl_f, &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
cleanup:
  if (!clingo_backend_end(back)) return false;
  return true;
}

void join_ad_choice(char *dst, char **src, size_t n) {
  *dst++ = '{';
  for (size_t i = 0; i < n; ++i, ++dst) {
    for (char *t = src[i]; *t; ++t) *dst++ = *t;
    *dst = ';';
  }
  *(dst-1) = '}'; dst[0] = '='; dst[1] = '1'; dst[2] = '.'; dst[3] = '\0';
}

bool join_ad_choice_sym(char *dst, size_t max_size, clingo_symbol_t *S, size_t n) {
  *dst++ = '{';
  for (size_t i = 0; i < n; ++i, ++dst) {
    if (!clingo_symbol_to_string(S[i], dst, max_size)) return false;
    dst += strlen(dst); *dst = ';';
  }
  *(dst-1) = '}'; dst[0] = '='; dst[1] = '1'; dst[2] = '.'; dst[3] = '\0';
  return true;
}

bool add_all_atoms_as_choice(clingo_control_t *C, program_t *P) {
  bool ok = false;
  clingo_backend_t *back;
  clingo_atom_t *heads = NULL;
  size_t nheads = P->PF_n + P->CF_n + P->NR_n;

  /* Get the control's backend. */
  if (!clingo_control_backend(C, &back)) return false;
  /* Startup the backend. */
  if (!clingo_backend_begin(back)) goto cleanup;

  /* Collect all probabilistic facts. */
  heads = (clingo_atom_t*) malloc(nheads*sizeof(clingo_atom_t));
  if (!heads) goto cleanup;
  size_t i_head = 0;
  for (size_t i = 0; i < P->PF_n; ++i)
    if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &heads[i_head++])) goto cleanup;
  for (size_t i = 0; i < P->CF_n; ++i)
    if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &heads[i_head++])) goto cleanup;
  for (size_t i = 0; i < P->NR_n; ++i)
    for (size_t j = 0; j < P->NR[i].n; ++j)
      if (!clingo_backend_add_atom(back, &P->NR[i].H[j], &heads[i_head++])) goto cleanup;

  /* Add them to the backend as a single choice. */
  if (!clingo_backend_rule(back, true, heads, nheads, NULL, 0)) goto cleanup;

  /* We're done with backend. */
  if (!clingo_backend_end(back)) { back = NULL; goto cleanup; }
  back = NULL;

  /* Add each AD as a choice with cardinality constraint of one. */
  char ad_choices[2048];
  for (size_t i = 0; i < P->AD_n; ++i) {
    join_ad_choice(ad_choices, (char**) P->AD[i].F, P->AD[i].n);
    if (!clingo_control_add(C, "base", NULL, 0, ad_choices)) goto cleanup;
  }
  /* Add neural ADs the same way. */
  for (size_t i = 0; i < P->NA_n; ++i)
    for (size_t j = 0; j < P->NA[i].n; ++j) {
      if (!join_ad_choice_sym(ad_choices, 2048, P->NA[i].H+j*P->NA[i].v, P->NA[i].v)) goto cleanup;
      if (!clingo_control_add(C, "base", NULL, 0, ad_choices)) goto cleanup;
    }
  ok = true;
cleanup:
  free(heads);
  /* Cleanup backend. */
  if (back) if (!clingo_backend_end(back)) return false;
  return ok;
}

bool add_neural_rule_atoms(clingo_backend_t *back, program_t *P, total_choice_t *theta) {
  clingo_atom_t h;
  clingo_literal_t B[64];
  size_t c = P->CF_n + P->PF_n;
  for (size_t i = 0; i < P->NR_n; ++i)
    for (size_t j = 0; j < P->NR[i].n; ++j) {
      /* Record body literals. */
      for (size_t b = 0; b < P->NR[i].k; ++b) {
        size_t u = j*P->NR[i].k+b;
        if (!clingo_backend_add_atom(back, &P->NR[i].B[u], (clingo_atom_t*) &B[b]))
          return false;
        if (!P->NR[i].S[u]) B[b] = -B[b];
      }
      /* Select head from outcomes. */
      for (size_t o = 0; o < P->NR[i].o; ++o) {
        if (!CHOICE_IS_TRUE(theta, c++)) continue;
        if (!clingo_backend_add_atom(back, &P->NR[i].H[j*P->NR[i].o+o], &h)) return false;
        /* Add neural rule. */
        if (!clingo_backend_rule(back, false, &h, 1, B, P->NR[i].k)) return false;
      }
    }
  return true;
}

bool add_neural_ad_atoms(clingo_backend_t *back, program_t *P, total_choice_t *theta) {
  clingo_atom_t h;
  clingo_literal_t B[64];
  size_t r = P->AD_n;
  for (size_t i = 0; i < P->NA_n; ++i)
    for (size_t j = 0; j < P->NA[i].n; ++j) {
      for (size_t b = 0; b < P->NA[i].k; ++b) {
        size_t u = j*P->NA[i].k+b;
        if (!clingo_backend_add_atom(back, &P->NA[i].B[u], (clingo_atom_t*) &B[b]))
          return false;
        if (!P->NA[i].S[u]) B[b] = -B[b];
      }
      for (size_t o = 0; o < P->NA[i].o; ++o) {
        if (!clingo_backend_add_atom(back, &P->NA[i].H[j*P->NA[i].v*P->NA[i].o +
              o*P->NA[i].v + theta->theta_ad[r++]], &h))
          return false;
        if (!clingo_backend_rule(back, false, &h, 1, B, P->NA[i].k)) return false;
      }
    }
  return true;
}

bool add_atoms_from_total_choice(clingo_control_t *C, program_t *P, total_choice_t *theta) {
  bool ok = false;
  clingo_backend_t *back;
  /* Get the control's backend. */
  if (!clingo_control_backend(C, &back)) return false;
  /* Startup the backend. */
  if (!clingo_backend_begin(back)) goto cleanup;
  /* Add the credal facts according to the total rule. */
  for (size_t i = 0; i < P->CF_n; ++i) {
    clingo_atom_t a;
    if (!CHOICE_IS_TRUE(theta, i)) continue;
    if (!clingo_backend_add_atom(back, &P->CF[i].cl_f, &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Add the probabilistic facts according to the total rule. */
  for (size_t i = 0; i < P->PF_n; ++i) {
    clingo_atom_t a;
    if (!CHOICE_IS_TRUE(theta, i + P->CF_n)) continue;
    if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Add the neural rules according to the total rule. */
  if (!add_neural_rule_atoms(back, P, theta)) goto cleanup;
  /* Add the annotated disjunction rules according to the total rule encoded by theta_ad. */
  for (size_t i = 0; i < P->AD_n; ++i) {
    clingo_atom_t a;
    if (!clingo_backend_add_atom(back, &P->AD[i].cl_F[theta->theta_ad[i]], &a)) goto cleanup;
    if (!clingo_backend_rule(back, false, &a, 1, NULL, 0)) goto cleanup;
  }
  /* Add the neural annotated disjunction rules according to the total rule encoded by theta_ad. */
  if (!add_neural_ad_atoms(back, P, theta)) goto cleanup;
  ok = true;
cleanup:
  /* Cleanup backend. */
  if (!clingo_backend_end(back)) return false;
  return ok;
}

bool _prepare_control(clingo_control_t **C, program_t *P, total_choice_t *theta,
    const char *nmodels, bool parallelize_clingo, const char *append) {
  /* Create new clingo controller. */
  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, C)) return false;
  /* Config to enumerate all models. */
  if (!setup_config(*C, nmodels, false)) return false;
  /* Add the purely logical part. */
  if (!clingo_control_add(*C, "base", NULL, 0, P->P)) return false;
  if (append) if (!clingo_control_add(*C, "base", NULL, 0, append)) return false;
  /* Add grounded probabilistic rules. */
  if (P->gr_P[0]) if (!clingo_control_add(*C, "base", NULL, 0, P->gr_P)) return false;
  if (!add_atoms_from_total_choice(*C, P, theta)) return false;
  return true;
}

bool prepare_control_preground(clingo_control_t **C, program_t *P, total_choice_t *theta,
    const char *nmodels, bool parallelize_clingo, const char *append, array_prob_fact_t *gr_PF,
    total_choice_t *gr_theta) {
  if (!_prepare_control(C, P, theta, nmodels, parallelize_clingo, append)) return false;
  if (!add_facts_from_total_choice(*C, gr_PF, gr_theta)) return false;
  /* Ground atoms. */
  if (!atomic_ground(*C, NULL, NULL)) return false;
  return true;
}

bool atomic_ground(clingo_control_t *C, clingo_ground_callback_t gcb, void *gdata) {
  static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  bool ok;
  pthread_mutex_lock(&mu);
  ok = clingo_control_ground(C, GROUND_DEFAULT_PARTS, 1, gcb, gdata);
  pthread_mutex_unlock(&mu);
  return ok;
}

bool prepare_control(clingo_control_t **C, program_t *P, total_choice_t *theta,
    const char *nmodels, bool parallelize_clingo, const char *append) {
  if (!_prepare_control(C, P, theta, nmodels, parallelize_clingo, append)) return false;
  if (!clingo_control_ground(*C, GROUND_DEFAULT_PARTS, 1, NULL, NULL)) return false;
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

bool has_total_model(program_t *P, total_choice_t *theta, bool *has) {
  clingo_control_t *C = NULL;
  clingo_solve_handle_t *handle;
  clingo_solve_result_bitset_t res;
  /* Prepare control according to the stable semantics. */
  if (!prepare_control(&C, P->stable, theta, "1", false, NULL)) goto cleanup;
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


