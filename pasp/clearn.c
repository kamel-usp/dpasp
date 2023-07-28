#include "clearn.h"

#include "carray.h"
#include "cdata.h"
#include "cground.h"

#include "../progressbar/progressbar.h"

static inline void update_ground_pr(program_t *P, prob_storage_t *Q) {
  for (size_t i = 0; i < Q->pr; ++i)
    for (size_t j = 0; j < Q->I_GR[i].n; ++j)
      P->PF[Q->I_GR[i].d[j]].p = P->PR[Q->I_PR[i]].p;
}

static inline bool forward_neural(program_t *P, observations_t *O) {
  for (size_t j = 0; j < P->NR_n; ++j)
    if (!update_forward_neural_rule(&P->NR[j], O->i, O->i+O->n)) return false;
  for (size_t j = 0; j < P->NA_n; ++j)
    if (!update_forward_neural_annot_disj(&P->NA[j], O->i, O->i+O->n)) return false;
  return true;
}

static inline bool backward_neural(program_t *P, prob_storage_t *Q) {
  for (size_t i_nr = 0; i_nr < Q->nr; ++i_nr)
    if (!backward_neural_rule(&P->NR[Q->I_NR[i_nr]])) return false;
  for (size_t i_na = 0; i_na < Q->na; ++i_na)
    if (!backward_neural_annot_disj(&P->NA[Q->I_NA[i_na]])) return false;
  return true;
}

void compute_fixpoint(program_t *P, prob_storage_t *Q, size_t N, double eta,
    PyArrayObject *obs_counts, observations_t *O) {
  /* Learning rule by soft-max is:
   *
   *   P(t = i) = (1/|O|) * sum_{o in O} P(t = i, O)/P(O)
   *
   */

  /* Reset probabilistic facts. */
  for (size_t i_pf = 0; i_pf < Q->n; ++i_pf) P->PF[Q->I_F[i_pf]].p = 0;
  /* Reset annotated disjunctions. */
  for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
    annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
    for (size_t j = 0; j < AD->n; ++j) AD->P[j] = 0;
  }
  /* Reset probabilistic rules. */
  for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr) P->PR[Q->I_PR[i_pr]].p = 0;

  /* Update parameters. */
  for (size_t i_o = 0; i_o < O->n; ++i_o) {
    prob_obs_storage_t *W = &Q->P[i_o];
    int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));
    /* Update probabilistic facts. */
    for (size_t i_pf = 0; i_pf < Q->n; ++i_pf) {
      /* P(t = i, O) = W->F[i_pf][1] */
      /* P(O)        = W->o          */
      P->PF[Q->I_F[i_pf]].p += c*(W->F[i_pf][1]/W->o);
    }
    /* Update annotated disjunctions. */
    for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
      /* P(t = i, O) = W->A[i_ad][j] */
      /* P(O)        = W->o          */
      annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
      for (size_t j = 0; j < AD->n; ++j)
        AD->P[j] += c*(W->A[i_ad][j]/W->o);
    }
    /* Update probabilistic rules. */
    for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr)
      P->PR[Q->I_PR[i_pr]].p += c*(W->R[i_pr][1]/W->o);
  }

  /* Divide probabilistic facts by the number of observations N. */
  for (size_t i_pf = 0; i_pf < Q->n; ++i_pf) P->PF[Q->I_F[i_pf]].p /= N;
  /* Divide annotated disjunctions by the number of observations N. */
  for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
    annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
    for (size_t j = 0; j < AD->n; ++j)
      AD->P[j] /= N;
  }
  /* Divide probabilistic rules by the number of observations N and number of grounded rules. */
  for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr) P->PR[Q->I_PR[i_pr]].p /= N*Q->I_GR[i_pr].n;
}

void compute_lagrange(program_t *P, prob_storage_t *Q, size_t N, double eta,
    PyArrayObject *obs_counts, observations_t *O) {
  /* Update parameters. */
  for (size_t i_o = 0; i_o < O->n; ++i_o) {
    prob_obs_storage_t *W = &Q->P[i_o];
    int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));

    /* Update probabilistic facts. */
    for (size_t i_pf = 0; i_pf < Q->n; ++i_pf)
      P->PF[Q->I_F[i_pf]].p += eta*c*(((W->F[i_pf][1] - W->F[i_pf][0])*0.5)/W->o);
    /* Update probabilistic rules with shared parameters. */
    for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr)
      P->PR[Q->I_PR[i_pr]].p += eta*c*(((W->R[i_pr][1] - W->R[i_pr][0])*0.5)/W->o);
    /* Update annotated disjunctions. */
    for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
      annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
      double dP = 0.0;
      for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
      for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*c*((W->A[i_ad][j] - dP/AD->n)/W->o);
    }
    /* Accumulate neural rule derivatives. */
    for (size_t i_nr = 0; i_nr < Q->nr; ++i_nr) {
      neural_rule_t *R = &P->NR[i_nr];
      for (size_t g = 0; g < R->n; ++g)
        for (size_t o = 0; o < R->o; ++o) {
          size_t u = g*2*R->o + o*2;
          P->NR[Q->I_NR[i_nr]].dw[i_o*R->o + g*R->o*P->batch + o] = eta*c*((W->NR[i_nr][u+1] - W->NR[i_nr][u])*0.5)/W->o;
        }
    }
    /* Accumulate neural annotated disjunction derivatives. */
    for (size_t i_na = 0; i_na < Q->na; ++i_na) {
      neural_annot_disj_t *A = &P->NA[Q->I_NA[i_na]];
      for (size_t g = 0; g < A->n; ++g)
        for (size_t o = 0; o < A->o; ++o) {
          double *derivs = W->NA[i_na] + g*A->v*A->o + o*A->v;
          size_t offset = i_o*A->o*A->v + g*A->o*A->v*P->batch + o*A->v;
          double dP = 0.0;
          for (size_t j = 0; j < A->v; ++j) dP += derivs[j];
          for (size_t j = 0; j < A->v; ++j) A->dw[offset + j] = eta*c*(derivs[j] - dP/A->v)/W->o;
        }
    }
  }
}

void compute_neurasp(program_t *P, prob_storage_t *Q, size_t N, double eta,
    PyArrayObject *obs_counts, observations_t *O) {
  /* Update parameters. */
  for (size_t i_o = 0; i_o < O->n; ++i_o) {
    prob_obs_storage_t *W = &Q->P[i_o];
    int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));

    /* Update probabilistic facts. */
    for (size_t i_pf = 0; i_pf < Q->n; ++i_pf)
      P->PF[Q->I_F[i_pf]].p += eta*c*((W->F[i_pf][1] - W->F[i_pf][0])/W->o);
    /* Update annotated disjunctions. */
    for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
      annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
      double dP = 0.0;
      for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
      for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*c*((2*W->A[i_ad][j] - dP)/W->o);
    }
  }
}

bool learn(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat, size_t which,
    uint8_t display) {
  observations_t O = {0}; /* Observations as a C type. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  size_t num_procs = 0, N = 0;
  bool ok = false;
  void (*alg[3])(program_t*, prob_storage_t*, size_t, double, PyArrayObject*, observations_t*) = {
    compute_fixpoint, compute_lagrange, compute_neurasp
  };
  bool derive = which != ALG_FIXPOINT;
  double ll = -INFINITY;
  progressbar *bar = display ? progressbar_new("Learning", niters,
                                               display == DISPLAY_LOGLIKELIHOOD) : NULL;

  if (display && !bar) goto cleanup;
  /* Reuse display for figuring if LL should be displayed. */
  display = (display == DISPLAY_LOGLIKELIHOOD);

  if (!init_observations(&O, obs, atoms)) goto cleanup;
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  if (needs_ground(P)) if (!ground_all(P, Q)) goto cleanup;

  if (which == ALG_FIXPOINT) {
    /* Compute |O| for fixpoint. */
    for (size_t i = 0; i < O.n; ++i)
      N += (int) *((int*) PyArray_GETPTR1(obs_counts, i));
  }

  for (size_t i = 0; i < niters; ++i) {
    P->batch = O.n;
    if (!forward_neural(P, &O)) goto cleanup;

    /* Compute probabilities. */
    if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, derive)) goto cleanup;

    alg[which](P, &Q[0], N, eta, obs_counts, &O);

    /* Update shared ground PFs from PRs. */
    update_ground_pr(P, &Q[0]);

    /* Update progress bar. */
    if (bar) progressbar_inc(bar, display ? (ll = ll_prob_storage_counts(&Q[0], O.n, obs_counts)) : 0.);

    /* Check for signals. */
    if (PyErr_CheckSignals()) goto cleanup;
  }

  if (!update_program_parameters(P, &Q[0])) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  if (bar) progressbar_finish(bar, ll);
  free_observations_contents(&O);
  for (size_t i = 1; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_prob_storage_contents(&Q[0], true);
  return ok;
}

bool learn_fixpoint(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, bool lstable_sat, uint8_t display) {
  return learn(P, obs, obs_counts, atoms, niters, 0., lstable_sat, ALG_FIXPOINT, display);
}

bool learn_lagrange(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat, uint8_t display) {
  return learn(P, obs, obs_counts, atoms, niters, eta, lstable_sat, ALG_LAGRANGE, display);
}

bool learn_neurasp(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat, uint8_t display) {
  return learn(P, obs, obs_counts, atoms, niters, eta, lstable_sat, ALG_NEURASP, display);
}

void compute_fixpoint_batch(program_t *P, prob_storage_t *Q, observations_t *O, double eta,
    double smooth) {
  /* Reset probabilistic facts. */
  for (size_t i_pf = 0; i_pf < Q->n; ++i_pf) P->PF[Q->I_F[i_pf]].p = 0;
  /* Reset annotated disjunctions. */
  for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
    annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
    for (size_t j = 0; j < AD->n; ++j) AD->P[j] = 0;
  }
  /* Reset probabilistic rules. */
  for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr) P->PR[Q->I_PR[i_pr]].p = 0;

  /* Update parameters. */
  for (size_t i_o = 0; i_o < O->n; ++i_o) {
    prob_obs_storage_t *W = &Q->P[i_o];
    /* Update probabilistic facts. */
    for (size_t i_pf = 0; i_pf < Q->n; ++i_pf) {
      /* P(t = i, O) = W->F[i_pf][1] */
      /* P(O)        = W->o          */
      P->PF[Q->I_F[i_pf]].p += W->F[i_pf][1]/W->o;
    }
    /* Update annotated disjunctions. */
    for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
      /* P(t = i, O) = W->A[i_ad][j] */
      /* P(O)        = W->o          */
      annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
      for (size_t j = 0; j < AD->n; ++j)
        AD->P[j] += W->A[i_ad][j]/W->o;
    }
    /* Update probabilistic rules. */
    for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr)
      P->PR[Q->I_PR[i_pr]].p += W->R[i_pr][1]/W->o;
  }

  /* Divide probabilistic facts by the number of observations N. */
  for (size_t i_pf = 0; i_pf < Q->n; ++i_pf) P->PF[Q->I_F[i_pf]].p /= O->n;
  /* Divide annotated disjunctions by the number of observations N. */
  for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
    annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
    for (size_t j = 0; j < AD->n; ++j)
      AD->P[j] /= O->n;
  }
  /* Divide probabilistic rules by the number of observations N and number of grounded rules. */
  for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr) P->PR[Q->I_PR[i_pr]].p /= O->n*Q->I_GR[i_pr].n;
}

void compute_lagrange_batch(program_t *P, prob_storage_t *Q, observations_t *O, double eta,
    double smooth) {
  /* Update parameters. */
  for (size_t i_o = 0; i_o < O->n; ++i_o) {
    prob_obs_storage_t *W = &Q->P[i_o];
    double p_o = W->o + smooth;

    /* Update probabilistic facts. */
    for (size_t i_pf = 0; i_pf < Q->n; ++i_pf)
      P->PF[Q->I_F[i_pf]].p += eta*(((W->F[i_pf][1] - W->F[i_pf][0])*0.5 + smooth)/p_o);
    /* Update annotated disjunctions. */
    for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
      annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
      double dP = 0.0;
      for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
      for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*((W->A[i_ad][j] - dP/AD->n + smooth)/p_o);
    }
    /* Update probabilistic rules with shared parameters. */
    for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr)
      P->PR[Q->I_PR[i_pr]].p += eta*(((W->R[i_pr][1] - W->R[i_pr][0])*0.5 + smooth)/p_o);
    /* Accumulate neural rule derivatives. */
    for (size_t i_nr = 0; i_nr < Q->nr; ++i_nr) {
      neural_rule_t *R = &P->NR[i_nr];
      for (size_t g = 0; g < R->n; ++g)
        for (size_t o = 0; o < R->o; ++o) {
          size_t u = g*2*R->o + o*2;
          P->NR[Q->I_NR[i_nr]].dw[i_o*R->o + g*R->o*P->batch + o] = eta*((W->NR[i_nr][u+1] - W->NR[i_nr][u])*0.5 + smooth)/p_o;
        }
    }
    /* Accumulate neural annotated disjunction derivatives. */
    for (size_t i_na = 0; i_na < Q->na; ++i_na) {
      neural_annot_disj_t *A = &P->NA[Q->I_NA[i_na]];
      for (size_t g = 0; g < A->n; ++g)
        for (size_t o = 0; o < A->o; ++o) {
          double *derivs = W->NA[i_na] + g*A->v*A->o + o*A->v;
          size_t offset = i_o*A->o*A->v + g*A->o*A->v*P->batch + o*A->v;
          double dP = 0.0;
          for (size_t j = 0; j < A->v; ++j) dP += derivs[j];
          for (size_t j = 0; j < A->v; ++j) A->dw[offset + j] = eta*((derivs[j] - dP/A->v)+smooth)/p_o;
        }
    }
  }
}

void compute_neurasp_batch(program_t *P, prob_storage_t *Q, observations_t *O, double eta,
    double smooth) {
  /* Update parameters. */
  for (size_t i_o = 0; i_o < O->n; ++i_o) {
    prob_obs_storage_t *W = &Q->P[i_o];
    double p_o = W->o + smooth;

    /* Update probabilistic facts. */
    for (size_t i_pf = 0; i_pf < Q->n; ++i_pf)
      P->PF[Q->I_F[i_pf]].p += eta*((W->F[i_pf][1] - W->F[i_pf][0] + smooth)/p_o);
    /* Update annotated disjunctions. */
    for (size_t i_ad = 0; i_ad < Q->m; ++i_ad) {
      annot_disj_t *AD = &P->AD[Q->I_A[i_ad]];
      double dP = 0.0;
      for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
      for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*((W->A[i_ad][j] - dP + smooth)/p_o);
    }
    /* Update probabilistic rules with shared parameters. */
    for (size_t i_pr = 0; i_pr < Q->pr; ++i_pr)
      P->PR[Q->I_PR[i_pr]].p += eta*((W->R[i_pr][1] - W->R[i_pr][0] + smooth)/p_o);
    /* Accumulate neural rule derivatives. */
    for (size_t i_nr = 0; i_nr < Q->nr; ++i_nr) {
      neural_rule_t *R = &P->NR[i_nr];
      for (size_t g = 0; g < R->n; ++g)
        for (size_t o = 0; o < R->o; ++o) {
          size_t u = g*2*R->o + o*2;
          P->NR[Q->I_NR[i_nr]].dw[i_o*R->o + g*R->o*P->batch + o] = eta*(W->NR[i_nr][u+1] - W->NR[i_nr][u] + smooth)/p_o;
        }
    }
    /* Accumulate neural annotated disjunction derivatives. */
    for (size_t i_na = 0; i_na < Q->na; ++i_na) {
      neural_annot_disj_t *A = &P->NA[Q->I_NA[i_na]];
      for (size_t g = 0; g < A->n; ++g)
        for (size_t o = 0; o < A->o; ++o) {
          double *derivs = W->NA[i_na] + g*A->v*A->o + o*A->v;
          size_t offset = i_o*A->o*A->v + g*A->o*A->v*P->batch + o*A->v;
          double dP = 0.0;
          for (size_t j = 0; j < A->v; ++j) dP += derivs[j];
          for (size_t j = 0; j < A->v; ++j) A->dw[offset + j] = eta*(2*derivs[j] - dP/A->v + smooth)/p_o;
        }
    }
  }
}

bool learn_batch(program_t *P, PyArrayObject *obs, size_t niters, double eta, size_t batch,
    double smooth, bool lstable_sat, size_t which, uint8_t display) {
  observations_t O = {0}; /* Dense representation of observations. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  size_t num_procs = 0;
  bool ok = false;
  void (*alg[3])(program_t*, prob_storage_t*, observations_t*, double, double) = {
    compute_fixpoint_batch, compute_lagrange_batch, compute_neurasp_batch
  };
  bool derive = which != ALG_FIXPOINT;
  size_t num_obs = PyArray_DIM(obs, 0);
  progressbar *bar = display ? progressbar_new("Learning", niters*(num_obs/batch),
                                               display == DISPLAY_LOGLIKELIHOOD) : NULL;
  double ll = -INFINITY;

  if (display && !bar) goto cleanup;
  /* Reuse display for figuring if LL should be displayed. */
  display = (display == DISPLAY_LOGLIKELIHOOD);

  if (!init_dense_observations(&O, obs, batch)) goto cleanup;
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  if (needs_ground(P)) if (!ground_all(P, Q)) goto cleanup;

  for (size_t i = 0; i < niters; ++i) {
    do {
      P->batch = O.n;
      if (!forward_neural(P, &O)) goto cleanup;

      /* Compute probabilities. */
      if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, derive)) goto cleanup;

      alg[which](P, &Q[0], &O, eta, smooth);

      /* Update shared ground PFs from PRs. */
      update_ground_pr(P, &Q[0]);

      /* Backpropagate neural components. */
      if (!backward_neural(P, &Q[0])) goto cleanup;

      /* Update progress bar. */
      if (bar) progressbar_inc(bar, display ? (ll = ll_prob_storage(&Q[0], O.n)/O.n) : 0.);

      if (!next_dense_observations(&O, obs)) goto cleanup;

      /* Check for signals. */
      if (PyErr_CheckSignals()) goto cleanup;
    } while (O.i);
  }

  if (!update_program_parameters(P, &Q[0])) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  if (bar) progressbar_finish(bar, ll);
  free_dense_observations_contents(&O);
  for (size_t i = 1; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_prob_storage_contents(&Q[0], true);
  return ok;
}

bool learn_fixpoint_batch(program_t *P, PyArrayObject *obs, size_t niters, size_t batch,
    double smooth, bool lstable_sat, uint8_t display) {
  return learn_batch(P, obs, niters, 0., batch, smooth, lstable_sat, ALG_FIXPOINT, display);
}

bool learn_lagrange_batch(program_t *P, PyArrayObject *obs, size_t niters, double eta, size_t batch,
    double smooth, bool lstable_sat, uint8_t display) {
  return learn_batch(P, obs, niters, eta, batch, smooth, lstable_sat, ALG_LAGRANGE, display);
}

bool learn_neurasp_batch(program_t *P, PyArrayObject *obs, size_t niters, double eta, size_t batch,
    double smooth, bool lstable_sat, uint8_t display) {
  return learn_batch(P, obs, niters, eta, batch, smooth, lstable_sat, ALG_NEURASP, display);
}

bool update_program_parameters(program_t *P, prob_storage_t *Q) {
  for (size_t i = 0; i < Q->n; ++i) {
    prob_fact_t *pf = &P->PF[Q->I_F[i]];
    PyObject *py_pf = pf->self;
    PyObject *p = PyFloat_FromDouble(pf->p);

    if (!p) {
      PyErr_SetString(PyExc_TypeError, "could not create a Python double from C double!");
      return false;
    }
    if (PyObject_SetAttrString(py_pf, "p", p) < 0) {
      PyErr_SetString(PyExc_AttributeError, "could not update PF.p!");
      Py_DECREF(p);
      return false;
    }
  }

  for (size_t i = 0; i < Q->m; ++i) {
    annot_disj_t *ad = &P->AD[Q->I_A[i]];
    PyObject *py_ad = ad->self;
    PyObject *py_ad_P = PyObject_GetAttrString(py_ad, "P");

    if (!py_ad_P) {
      PyErr_SetString(PyExc_AttributeError, "could not retrieve AD.P!");
      return false;
    }

    for (size_t j = 0; j < ad->n; ++j) {
      PyObject *p = PyFloat_FromDouble(ad->P[j]);
      if (!p) {
        PyErr_SetString(PyExc_TypeError, "could not create a Python double from C double!");
        Py_DECREF(py_ad_P);
        return false;
      }
      if (PyList_SetItem(py_ad_P, j, p) < 0) {
        Py_DECREF(p);
        Py_DECREF(py_ad_P);
        return false;
      }
    }

    Py_DECREF(py_ad_P);
  }

  for (size_t i = 0; i < Q->pr; ++i) {
    prob_rule_t *pr = &P->PR[Q->I_PR[i]];
    PyObject *py_pr = pr->self;
    PyObject *p = PyFloat_FromDouble(pr->p);

    if (!p) {
      PyErr_SetString(PyExc_TypeError, "could not create a Python double from C double!");
      return false;
    }
    if (PyObject_SetAttrString(py_pr, "p", p) < 0) {
      PyErr_SetString(PyExc_AttributeError, "could not update PR.p!");
      Py_DECREF(p);
      return false;
    }
  }

  return true;
}
