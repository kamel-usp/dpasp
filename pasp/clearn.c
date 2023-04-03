#include "clearn.h"

#include "carray.h"
#include "cdata.h"
#include "cground.h"

void update_ground_pr(program_t *P, prob_storage_t *Q) {
  for (size_t i = 0; i < Q->pr; ++i)
    for (size_t j = 0; j < Q->I_GR[i].n; ++j)
      P->PF[Q->I_GR[i].d[j]].p = P->PR[Q->I_PR[i]].p;
}

bool learn_fixpoint(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, bool lstable_sat) {
  observations_t O = {0}; /* Observations as a C type. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  size_t num_procs = 0, N = 0;
  bool ok = false;

  if (!init_observations(&O, obs, atoms)) goto cleanup;
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  if (needs_ground(P)) if (!ground_all(P, Q)) goto cleanup;

  /* Compute |O|. */
  for (size_t i = 0; i < O.n; ++i)
    N += (int) *((int*) PyArray_GETPTR1(obs_counts, i));

  for (size_t i = 0; i < niters; ++i) {
    /* Compute probabilities. */
    if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, false)) goto cleanup;

    /* Learning rule by soft-max is:
     *
     *   P(t = i) = (1/|O|) * sum_{o in O} P(t = i, O)/P(O)
     *
     */

    /* Reset probabilistic facts. */
    for (size_t i_pf = 0; i_pf < Q[0].n; ++i_pf) P->PF[Q[0].I_F[i_pf]].p = 0;
    /* Reset annotated disjunctions. */
    for (size_t i_ad = 0; i_ad < Q[0].m; ++i_ad) {
      annot_disj_t *AD = &P->AD[Q[0].I_A[i_ad]];
      for (size_t j = 0; j < AD->n; ++j) AD->P[j] = 0;
    }
    /* Reset probabilistic rules. */
    for (size_t i_pr = 0; i_pr < Q[0].pr; ++i_pr) P->PR[Q[0].I_PR[i_pr]].p = 0;

    /* Update parameters. */
    for (size_t i_o = 0; i_o < O.n; ++i_o) {
      prob_obs_storage_t *W = &Q[0].P[i_o];
      int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));
      /* Update probabilistic facts. */
      for (size_t i_pf = 0; i_pf < Q[0].n; ++i_pf) {
        /* P(t = i, O) = W->F[i_pf][1] */
        /* P(O)        = W->o          */
        P->PF[Q[0].I_F[i_pf]].p += c*(W->F[i_pf][1]/W->o);
      }
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < Q[0].m; ++i_ad) {
        /* P(t = i, O) = W->A[i_ad][j] */
        /* P(O)        = W->o          */
        annot_disj_t *AD = &P->AD[Q[0].I_A[i_ad]];
        for (size_t j = 0; j < AD->n; ++j)
          AD->P[j] += c*(W->A[i_ad][j]/W->o);
      }
      /* Update probabilistic rules. */
      for (size_t i_pr = 0; i_pr < Q[0].pr; ++i_pr)
        P->PR[Q[0].I_PR[i_pr]].p += c*(W->R[i_pr][1]/W->o);
    }

    /* Divide probabilistic facts by the number of observations N. */
    for (size_t i_pf = 0; i_pf < Q[0].n; ++i_pf) P->PF[Q[0].I_F[i_pf]].p /= N;
    /* Divide annotated disjunctions by the number of observations N. */
    for (size_t i_ad = 0; i_ad < Q[0].m; ++i_ad) {
      annot_disj_t *AD = &P->AD[Q[0].I_A[i_ad]];
      for (size_t j = 0; j < AD->n; ++j)
        AD->P[j] /= N;
    }
    /* Divide probabilistic rules by the number of observations N and number of grounded rules. */
    for (size_t i_pr = 0; i_pr < Q[0].pr; ++i_pr) P->PR[Q[0].I_PR[i_pr]].p /= N*Q[0].I_GR[i_pr].n;
    /* Update shared ground PFs from PRs. */
    update_ground_pr(P, &Q[0]);

    /* Check for signals. */
    if (PyErr_CheckSignals()) goto cleanup;
  }

  if (!update_program_parameters(P, &Q[0])) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_observations_contents(&O);
  for (size_t i = 1; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_prob_storage_contents(&Q[0], true);
  return ok;
}

bool learn_lagrange(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat) {
  observations_t O = {0}; /* Observations as a C type. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  size_t num_procs = 0;
  bool ok = false;

  if (!init_observations(&O, obs, atoms)) goto cleanup;
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  if (needs_ground(P)) if (!ground_all(P, Q)) goto cleanup;

  for (size_t i = 0; i < niters; ++i) {
    /* Compute probabilities. */
    if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, true)) goto cleanup;

    /* Update parameters. */
    for (size_t i_o = 0; i_o < O.n; ++i_o) {
      prob_obs_storage_t *W = &Q[0].P[i_o];
      int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));

      /* Update probabilistic facts. */
      for (size_t i_pf = 0; i_pf < Q[0].n; ++i_pf)
        P->PF[Q[0].I_F[i_pf]].p += eta*c*(((W->F[i_pf][1] - W->F[i_pf][0])*0.5)/W->o);
      /* Update probabilistic rules with shared parameters. */
      for (size_t i_pr = 0; i_pr < Q[0].pr; ++i_pr)
        P->PR[Q[0].I_PR[i_pr]].p += eta*c*(((W->R[i_pr][1] - W->R[i_pr][0])*0.5)/W->o);
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < Q[0].m; ++i_ad) {
        annot_disj_t *AD = &P->AD[Q[0].I_A[i_ad]];
        double dP = 0.0;
        for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
        for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*c*((W->A[i_ad][j] - dP/AD->n)/W->o);
      }
    }

    /* Update shared ground PFs from PRs. */
    update_ground_pr(P, &Q[0]);

    /* Check for signals. */
    if (PyErr_CheckSignals()) goto cleanup;
  }

  if (!update_program_parameters(P, &Q[0])) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_observations_contents(&O);
  for (size_t i = 1; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_prob_storage_contents(&Q[0], true);
  return ok;
}

bool learn_lagrange_batch(program_t *P, PyArrayObject *obs, size_t niters, double eta,
    size_t batch, bool lstable_sat) {
  observations_t O = {0}; /* Dense representation of observations. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  size_t num_procs = 0;
  bool ok = false;

  if (!init_dense_observations(&O, obs, batch)) goto cleanup;
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  if (needs_ground(P)) if (!ground_all(P, Q)) goto cleanup;

  for (size_t i = 0; i < niters; ++i) {
    for (size_t j = 0; j < P->NR_n; ++j)
      if (!update_forward_neural_rule(&P->NR[j], O.i, O.i+O.n)) goto cleanup;
    for (size_t j = 0; j < P->NA_n; ++j)
      if (!update_forward_neural_annot_disj(&P->NA[j], O.i, O.i+O.n)) goto cleanup;

    /* Compute probabilities. */
    if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, true)) goto cleanup;

    /* Update parameters. */
    for (size_t i_o = 0; i_o < O.n; ++i_o) {
      prob_obs_storage_t *W = &Q[0].P[i_o];

      /* Update probabilistic facts. */
      for (size_t i_pf = 0; i_pf < Q[0].n; ++i_pf)
        P->PF[Q[0].I_F[i_pf]].p += eta*(((W->F[i_pf][1] - W->F[i_pf][0])*0.5)/W->o);
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < Q[0].m; ++i_ad) {
        annot_disj_t *AD = &P->AD[Q[0].I_A[i_ad]];
        double dP = 0.0;
        for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
        for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*((W->A[i_ad][j] - dP/AD->n)/W->o);
      }
      /* Update probabilistic rules with shared parameters. */
      for (size_t i_pr = 0; i_pr < Q[0].pr; ++i_pr)
        P->PR[Q[0].I_PR[i_pr]].p += eta*(((W->R[i_pr][1] - W->R[i_pr][0])*0.5)/W->o);
      /* Accumulate neural rule derivatives. */
      for (size_t i_nr = 0; i_nr < Q[0].nr; ++i_nr)
        P->NR[Q[0].I_NR[i_nr]].dw[i_o] = eta*((W->NR[i_nr][1] - W->NR[i_nr][0])*0.5)/W->o;
      /* Accumulate neural annotated disjunction derivatives. */
      for (size_t i_na = 0; i_na < Q[0].na; ++i_na) {
        neural_annot_disj_t *A = &P->NA[Q[0].I_NA[i_na]];
        float dP = 0.0;
        for (size_t j = 0; j < A->v; ++j) dP += W->NA[i_na][j];
        for (size_t j = 0; j < A->v; ++j) A->dw[i_o*A->v + j] = eta*(W->NA[i_na][j] - dP/A->v)/W->o;
      }
    }

    /* Update shared ground PFs from PRs. */
    update_ground_pr(P, &Q[0]);

    /* Backpropagate neural components. */
    for (size_t i_nr = 0; i_nr < Q[0].nr; ++i_nr)
      if (!backward_neural_rule(&P->NR[Q[0].I_NR[i_nr]], 0, O.n)) goto cleanup;
    for (size_t i_na = 0; i_na < Q[0].na; ++i_na)
      if (!backward_neural_annot_disj(&P->NA[Q[0].I_NA[i_na]], 0, O.n)) goto cleanup;

    if (!next_dense_observations(&O, obs)) goto cleanup;

    /* Check for signals. */
    if (PyErr_CheckSignals()) goto cleanup;
  }

  if (!update_program_parameters(P, &Q[0])) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_dense_observations_contents(&O);
  for (size_t i = 1; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_prob_storage_contents(&Q[0], true);
  return ok;
}

bool learn_neurasp(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat) {
  observations_t O = {0}; /* Observations as a C type. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  size_t num_procs = 0;
  bool ok = false;

  if (!init_observations(&O, obs, atoms)) goto cleanup;
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  if (needs_ground(P)) if (!ground_all(P, Q)) goto cleanup;

  for (size_t i = 0; i < niters; ++i) {
    /* Compute probabilities. */
    if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, true)) goto cleanup;

    /* Update parameters. */
    for (size_t i_o = 0; i_o < O.n; ++i_o) {
      prob_obs_storage_t *W = &Q[0].P[i_o];
      int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));

      /* Update probabilistic facts. */
      for (size_t i_pf = 0; i_pf < Q[0].n; ++i_pf)
        P->PF[Q[0].I_F[i_pf]].p += eta*c*((W->F[i_pf][1] - W->F[i_pf][0])/W->o);
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < Q[0].m; ++i_ad) {
        annot_disj_t *AD = &P->AD[Q[0].I_A[i_ad]];
        double dP = 0.0;
        for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
        for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*c*((2*W->A[i_ad][j] - dP)/W->o);
      }
    }

    /* Check for signals. */
    if (PyErr_CheckSignals()) goto cleanup;
  }

  if (!update_program_parameters(P, &Q[0])) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_observations_contents(&O);
  for (size_t i = 1; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_prob_storage_contents(&Q[0], true);
  return ok;
}

bool learn_neurasp_batch(program_t *P, PyArrayObject *obs, size_t niters, double eta,
    size_t batch, bool lstable_sat) {
  observations_t O = {0}; /* Dense representation of observations. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  size_t num_procs = 0;
  bool ok = false;

  if (!init_dense_observations(&O, obs, batch)) goto cleanup;
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  if (needs_ground(P)) if (!ground_all(P, Q)) goto cleanup;

  for (size_t i = 0; i < niters; ++i) {
    for (size_t j = 0; j < P->NR_n; ++j)
      if (!update_forward_neural_rule(&P->NR[j], O.i, O.i+O.n)) goto cleanup;
    for (size_t j = 0; j < P->NA_n; ++j)
      if (!update_forward_neural_annot_disj(&P->NA[j], O.i, O.i+O.n)) goto cleanup;

    /* Compute probabilities. */
    if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, true)) goto cleanup;

    /* Update parameters. */
    for (size_t i_o = 0; i_o < O.n; ++i_o) {
      prob_obs_storage_t *W = &Q[0].P[i_o];

      /* Update probabilistic facts. */
      for (size_t i_pf = 0; i_pf < Q[0].n; ++i_pf)
        P->PF[Q[0].I_F[i_pf]].p += eta*((W->F[i_pf][1] - W->F[i_pf][0])/W->o);
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < Q[0].m; ++i_ad) {
        annot_disj_t *AD = &P->AD[Q[0].I_A[i_ad]];
        double dP = 0.0;
        for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
        for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*((W->A[i_ad][j] - dP)/W->o);
      }
      /* Update probabilistic rules with shared parameters. */
      for (size_t i_pr = 0; i_pr < Q[0].pr; ++i_pr)
        P->PR[Q[0].I_PR[i_pr]].p += eta*((W->R[i_pr][1] - W->R[i_pr][0])/W->o);
      /* Accumulate neural rule derivatives. */
      for (size_t i_nr = 0; i_nr < Q[0].nr; ++i_nr)
        P->NR[Q[0].I_NR[i_nr]].dw[i_o] = eta*(W->NR[i_nr][1] - W->NR[i_nr][0])/W->o;
      /* Accumulate neural annotated disjunction derivatives. */
      for (size_t i_na = 0; i_na < Q[0].na; ++i_na) {
        neural_annot_disj_t *A = &P->NA[Q[0].I_NA[i_na]];
        float dP = 0.0;
        for (size_t j = 0; j < A->v; ++j) dP += W->NA[i_na][j];
        for (size_t j = 0; j < A->v; ++j) A->dw[i_o*A->v + j] = eta*(2*W->NA[i_na][j] - dP)/W->o;
      }
    }

    /* Update shared ground PFs from PRs. */
    update_ground_pr(P, &Q[0]);

    /* Backpropagate neural components. */
    for (size_t i_nr = 0; i_nr < Q[0].nr; ++i_nr)
      if (!backward_neural_rule(&P->NR[Q[0].I_NR[i_nr]], 0, O.n)) goto cleanup;
    for (size_t i_na = 0; i_na < Q[0].na; ++i_na)
      if (!backward_neural_annot_disj(&P->NA[Q[0].I_NA[i_na]], 0, O.n)) goto cleanup;

    if (!next_dense_observations(&O, obs)) goto cleanup;

    /* Check for signals. */
    if (PyErr_CheckSignals()) goto cleanup;
  }

  if (!update_program_parameters(P, &Q[0])) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_dense_observations_contents(&O);
  for (size_t i = 1; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_prob_storage_contents(&Q[0], true);
  return ok;
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
