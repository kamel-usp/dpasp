#include "caseo.h"

#include <stdio.h>
#include <math.h>

#include "cutils.h"

bool watch_minimize(clingo_weight_t p, const clingo_weighted_literal_t *W, size_t n, void *data) {
  void **pack = (void**) data;
  clingo_weighted_literal_t *wlits = (clingo_weighted_literal_t*) pack[0];
  size_t *i = (size_t*) pack[1];

  /* If we ever accept optimization inside dPASP, we may set p = 0 for ASEO and p > 0 for
   * regular optimization. */

  for (size_t j = 0; j < n; ++j) wlits[*i+j] = W[j];
  *i += n;

  return true;
}

bool control_set_nmodels(clingo_control_t *C, size_t n) {
  clingo_configuration_t *cfg = NULL;
  clingo_id_t cfg_root, cfg_sub;

  /* String to integer. */
#define MAX_N_STR 30
  char n_str[MAX_N_STR + 2];
  size_t i, d;
  n_str[MAX_N_STR+1] = '\0';
  for (i = 0, d = n; d > 9; d /= 10) n_str[MAX_N_STR-(i++)] = '0' + (d % 10);
  n_str[MAX_N_STR-i] = '0' + d;
  char *nmodels = n_str + MAX_N_STR - i;
#undef MAX_N_STR

  if (!clingo_control_configuration(C, &cfg)) return false;
  if (!clingo_configuration_root(cfg, &cfg_root)) return false;
  if (!clingo_configuration_map_at(cfg, cfg_root, "solve.models", &cfg_sub)) return false;
  if (!clingo_configuration_value_set(cfg, cfg_sub, nmodels)) return false;

  return true;
}

/** Probabilistic components to weak rules. */
bool pc2wr(program_t *P, clingo_control_t *C, clingo_backend_t *back, int scale,
    clingo_weighted_literal_t *W) {
  bool ok = false;
  clingo_weighted_literal_t wl = {0};
  clingo_atom_t choice;

  if (!clingo_backend_begin(back)) goto cleanup;

  for (size_t i = 0; i < P->PF_n; ++i) {
    if (!clingo_backend_add_atom(back, &P->PF[i].cl_f, &choice)) goto cleanup;
    wl.literal = choice;
    wl.weight = -round(scale*log(P->PF[i].p/(1-P->PF[i].p)));
    if (!clingo_backend_rule(back, true, &choice, 1, NULL, 0)) goto cleanup;
    if (!clingo_backend_minimize(back, 0, &wl, 1)) goto cleanup;
    W[i] = wl;
  }

  /* Clingo unfortunately does not support adding cardinality constraint bounds that are not lower
   * bounds through backend. So instead, we have to do string translation. Maybe handling the AST
   * might be more efficient, but the AST API is still a mistery to me. */
#define MAX_RULE_LENGTH 8192
  char rule[MAX_RULE_LENGTH];
  size_t offset;
  for (size_t i = 0; i < P->AD_n; ++i) {
    rule[0] = '{'; rule[1] = '\0';
    offset = 1;
    offset += sprintf(rule+offset, "%s", P->AD[i].F[0]);
    if (!clingo_backend_add_atom(back, &P->AD[i].cl_F[0], &choice)) goto cleanup;
    wl.literal = choice;
    wl.weight = -round(scale*log(P->AD[i].P[0]/(1-P->AD[i].P[0])));
    if (!clingo_backend_minimize(back, 0, &wl, 1)) goto cleanup;
    W[P->PF_n] = wl;
    for (size_t j = 1; j < P->AD[i].n; ++j) {
      offset += sprintf(rule+offset, "; %s", P->AD[i].F[j]);
      if (!clingo_backend_add_atom(back, &P->AD[i].cl_F[j], &choice)) goto cleanup;
      wl.literal = choice;
      wl.weight = -round(scale*log(P->AD[i].P[j]/(1-P->AD[i].P[j])));
      if (!clingo_backend_minimize(back, 0, &wl, 1)) goto cleanup;
      W[j+P->PF_n] = wl;
    }
    strcpy(rule+offset, "} = 1.");
    if (!clingo_control_add(C, "base", NULL, 0, rule)) goto cleanup;
  }
#undef MAX_RULE_LENGTH
  if (!clingo_backend_end(back)) goto cleanup;

  ok = true;
cleanup:
  return ok;
}

bool aseo_solve(program_t *P, clingo_control_t *C, size_t k,
    clingo_solve_result_bitset_t *solve_ret, size_t *N, models_t *models, observations_t *O,
    bool (*f)(const clingo_model_t*, program_t*, models_t*, size_t, observations_t*,
      clingo_control_t*)) {
  bool ok = false, opt;
  clingo_solve_handle_t *handle;
  const clingo_model_t *M;
  size_t m;

  if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle)) goto cleanup;
  for (m = 0; true;) {
    if (!clingo_solve_handle_resume(handle)) goto cleanup;
    if (!clingo_solve_handle_model(handle, &M)) goto cleanup;
    if (M) {
      /* Whether this model is optimal. */
      if (!clingo_model_optimality_proven(M, &opt)) goto cleanup;
      if (!opt) continue;
      ctree_t *leaf = ctree_add(&models->root, M, P);
      if (!leaf) goto cleanup;
      models->L[*N+m] = leaf;
      if (!f(M, P, models, *N+m, O, C)) goto cleanup;
      ++m;
      /* DEBUG BEGIN */
      /*printf("=================\n%lu-th model (run %lu):\n=================\n", *N+m, m);*/
      /*print_solution(M);*/
      /*int64_t cost;*/
      /*if (!clingo_model_cost(M, &cost, 1)) goto cleanup;*/
      /*printf("Cost: %ld\tOptimal: %d\n", cost, opt);*/
      /* DEBUG END */
    } else break;
  }
  if (!clingo_solve_handle_get(handle, solve_ret)) goto cleanup;

  *N += m;

  ok = true;
cleanup:
  return ok;
}

bool set_upper_bound(clingo_backend_t *back, clingo_weighted_literal_t *W, size_t n,
    clingo_weighted_literal_t *U, int cost) {
  /* We assume there is only one optimization level, so there is only one cost: some objective
   * function that is proportional to the log-likelihood. */
  int l = -cost;
  for (size_t i = 0; i < n; ++i)
    if (W[i].weight > 0) {
      l += W[i].weight;
      U[i].literal = -W[i].literal;
      U[i].weight = W[i].weight;
    } else {
      U[i].literal = W[i].literal;
      U[i].weight = -W[i].weight;
    }
  if (!clingo_backend_begin(back)) return false;
  bool ok = clingo_backend_weight_rule(back, false, NULL, 0, l, U, n);
  if (!clingo_backend_end(back)) return false;
  return ok;
}

models_t* aseo(program_t *P, size_t k, psemantics_t psem, observations_t *O, int scale,
    bool (*f)(const clingo_model_t*, program_t*, models_t*, size_t, observations_t*,
      clingo_control_t*)) {
  bool ok = false;
  clingo_control_t *C = NULL;
  clingo_backend_t *back = NULL;
  /*clingo_ground_program_observer_t obs = {NULL}; obs.minimize = watch_minimize;*/
  models_t *M = NULL;

  size_t n = num_prob_params(P);
  clingo_weighted_literal_t *W = (clingo_weighted_literal_t*) malloc(n*sizeof(clingo_weighted_literal_t));
  clingo_weighted_literal_t *U = (clingo_weighted_literal_t*) malloc(n*sizeof(clingo_weighted_literal_t));
  if (!(W && U)) goto nomem;
  M = O ? models_create(k, O->n, true) : models_create(k, P->Q_n, false);

  clingo_configuration_t *cfg = NULL;
  clingo_id_t cfg_root, cfg_sub;

  /* Create new clingo controller. */
  if (!clingo_control_new(NULL, 0, undef_atom_ignore, NULL, 20, &C)) return NULL;
  /* Set parallel_mode to "NUM_PROCS,compete". */
  if (!clingo_control_configuration(C, &cfg)) return NULL;
  if (!clingo_configuration_root(cfg, &cfg_root)) return NULL;
  if (!clingo_configuration_map_at(cfg, cfg_root, "solve.parallel_mode", &cfg_sub)) return NULL;
  if (!clingo_configuration_value_set(cfg, cfg_sub, NUM_PROCS_CONFIG_STR)) return NULL;
  /* Set optimization mode to enumeration of optimal models (optN). */
  if (!clingo_configuration_map_at(cfg, cfg_root, "solve.opt_mode", &cfg_sub)) return NULL;
  if (!clingo_configuration_value_set(cfg, cfg_sub, "optN")) return NULL;
  /* Add logical part. */
  if (!clingo_control_add(C, "base", NULL, 0, P->P)) return NULL;
  /* Add grounded part. */
  if (P->gr_P[0]) if (!clingo_control_add(C, "base", NULL, 0, P->gr_P)) return NULL;

  /*size_t i_W = 0;*/
  /*void *pack[] = {(void*) W, (void*) &i_W};*/
  if (!clingo_control_backend(C, &back)) goto cleanup;
  /* Convert probabilistic components into weak rules. */
  if (!pc2wr(P, C, back, scale, W)) goto cleanup;
  /*if (!clingo_control_register_observer(C, &obs, false, (void*) pack)) goto cleanup;*/
  if (!control_set_nmodels(C, k)) goto cleanup;
  if (!clingo_control_ground(C, GROUND_DEFAULT_PARTS, 1, NULL, NULL)) goto cleanup;

  const clingo_statistics_t *stats;
  uint64_t root_key, costs_key, cost_key;
  if (!clingo_control_statistics(C, &stats)) goto cleanup;
  if (!clingo_statistics_root(stats, &root_key)) goto cleanup;
  if (!clingo_statistics_map_at(stats, root_key, "summary.costs", &costs_key)) goto cleanup;

  clingo_solve_result_bitset_t res;

  size_t m = 0; /* Number of optimal models seen so far. */
  double cost;
  if (!aseo_solve(P, C, k, &res, &m, M, O, f)) goto cleanup;
  while ((res & clingo_solve_result_satisfiable) && !(res & clingo_solve_result_interrupted)) {
    if (m >= k) break;
    else if (!control_set_nmodels(C, k-m)) goto cleanup;

    /* If clingo does not update keys on solving, then we can push this line up to the init block. */
    if (!clingo_statistics_array_at(stats, costs_key, 0, &cost_key)) goto cleanup;
    if (!clingo_statistics_value_get(stats, cost_key, &cost)) goto cleanup;

    if (!set_upper_bound(back, W, n, U, (int) cost)) goto cleanup;
    if (!aseo_solve(P, C, k, &res, &m, M, O, f)) goto cleanup;
  }
  M->m = m;

  /* Debug: write models to dot. */
  char buffer[1048576];
  if (!ctree_dot(&M->root, P, buffer)) goto cleanup;
  FILE *file = fopen("/tmp/test.dot", "w");
  fputs(buffer, file);
  fclose(file);

  ok = true;
  goto cleanup;
nomem:
  PyErr_SetString(PyExc_MemoryError, "could not allocate enough memory for ASEO!");
cleanup:
  if (clingo_error_code() != clingo_error_success) raise_clingo_error(NULL);
  else if (!ok) PyErr_SetString(PyExc_RuntimeError, "an error has occurred during ASEO!");
  clingo_control_free(C);
  free(W); free(U);
  if (!ok) { models_free(M); M = NULL; }
  return M;
}
