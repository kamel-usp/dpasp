#include "clearn.h"

/*#include <wchar.h>*/
/*#include <locale.h>*/

#include "carray.h"
#include "cdata.h"

bool init_indices(indices_t *I, program_t *P) {
  size_t n, m, nr, na = nr = m = n = 0;
  uint16_t *I_F, *I_A, *I_NR, *I_NA = I_NR = I_A = I_F = NULL;
  uint16_t *O_NR, *O_NA = O_NR = NULL;

  for (size_t i = 0; i < P->PF_n; ++i) if (P->PF[i].learnable) ++n;
  if (n) {
    I_F = (uint16_t*) malloc(n*sizeof(uint16_t));
    if (!I_F) goto cleanup;
    for (size_t i, j = i = 0; i < P->PF_n; ++i) if (P->PF[i].learnable) I_F[j++] = i;
  }

  for (size_t i = 0; i < P->AD_n; ++i) if (P->AD[i].learnable) ++m;
  if (m) {
    I_A = (uint16_t*) malloc(m*sizeof(uint16_t));
    if (!I_A) goto cleanup;
    for (size_t i, j = i = 0; i < P->AD_n; ++i) if (P->AD[i].learnable) I_A[j++] = i;
  }

  for (size_t i = 0; i < P->NR_n; ++i) if (P->NR[i].learnable) ++nr;
  if (nr) {
    I_NR = (uint16_t*) malloc(nr*sizeof(uint16_t));
    if (!I_NR) goto cleanup;
    O_NR = (uint16_t*) malloc(nr*sizeof(uint16_t));
    if (!O_NR) goto cleanup;
    size_t s = P->PF_n;
    for (size_t i, j = i = 0; i < P->NR_n; ++i) {
      if (P->NR[i].learnable) { I_NR[j] = i; O_NR[j++] = s; }
      s += P->NR[i].n;
    }
  }

  for (size_t i = 0; i < P->NA_n; ++i) if (P->NA[i].learnable) ++na;
  if (na) {
    I_NA = (uint16_t*) malloc(na*sizeof(uint16_t));
    if (!I_NA) goto cleanup;
    O_NA = (uint16_t*) malloc(na*sizeof(uint16_t));
    if (!O_NA) goto cleanup;
    size_t s = P->AD_n;
    for (size_t i, j = i = 0; i < P->NA_n; ++i) {
      if (P->NA[i].learnable) { I_NA[j] = i; O_NA[j++] = s; }
      s += P->NA[i].n;
    }
  }

  I->n = n; I->m = m; I->nr = nr; I->na = na;
  I->F = I_F; I->A = I_A; I->NR = I_NR; I->NA = I_NA;
  I->O_NR = O_NR; I->O_NA = O_NA;

  return true;
cleanup:
  PyErr_SetString(PyExc_MemoryError, "could not allocate memory in init_indices!");
  free(I_F); free(I_A); free(I_NR); free(I_NA); free(O_NR); free(O_NA);
  return false;
}

void free_indices_contents(indices_t *I) { free(I->F); free(I->A); }

bool init_parameters(parameters_t *W, program_t *P) {
  indices_t I = {0};
  uint16_t *I_F, *I_A = I_F = NULL;
  double (*F)[2] = NULL;
  double **A = NULL;
  size_t n, m;

  if (!init_indices(&I, P)) return false;
  n = I.n; m = I.m; I_F = I.F; I_A = I.A;

  F = (double(*)[2]) malloc(n*sizeof(double[2]));
  if (!F) goto cleanup;
  A = (double**) malloc(m*sizeof(double*));
  if (!A) goto cleanup;

  for (size_t i = 0; i < m; ++i) {
    size_t c = P->AD[I_A[i]].n;
    A[i] = (double*) malloc(c*sizeof(double));
    if (!A[i]) {
      for (size_t j = 0; j < i; ++j) free(A[j]);
      goto cleanup;
    }
  }

  W->n = n; W->m = m;
  W->F = F; W->A = A;
  W->I_F = I_F; W->I_A = I_A;

  return true;
cleanup:
  PyErr_SetString(PyExc_MemoryError, "could not allocate memory in init_parameters!");
  free(I_F); free(I_A);
  free(F); free(A);
  return false;
}

void free_parameters_contents(parameters_t *W) {
  free(W->F);
  for (size_t i = 0; i < W->m; ++i) free(W->A[i]);
  free(W->A);
  free(W->I_F); free(W->I_A);
}

void free_parameters(parameters_t *W) { free_parameters_contents(W); free(W); }

bool learn_fixpoint(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, bool lstable_sat) {
  observations_t O = {0}; /* Observations as a C type. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  indices_t I = {0};
  size_t num_procs = 0, N = 0;
  bool ok = false;

  if (!init_observations(&O, obs, atoms)) goto cleanup;
  if (!init_indices(&I, P)) goto cleanup;
  Q[0].I_F = I.F; Q[0].n = I.n; Q[0].I_A = I.A; Q[0].m = I.m;
  if (!I.n && !I.m) {
    PyErr_SetString(PyExc_ValueError, "program is not learnable!");
    return false;
  }
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

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
    for (size_t i_pf = 0; i_pf < I.n; ++i_pf) P->PF[I.F[i_pf]].p = 0;
    /* Reset annotated disjunctions. */
    for (size_t i_ad = 0; i_ad < I.m; ++i_ad) {
      annot_disj_t *AD = &P->AD[I.A[i_ad]];
      for (size_t j = 0; j < AD->n; ++j) AD->P[j] = 0;
    }

    /* Update parameters. */
    for (size_t i_o = 0; i_o < O.n; ++i_o) {
      prob_obs_storage_t *W = &Q[0].P[i_o];
      int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));
      /* Update probabilistic facts. */
      for (size_t i_pf = 0; i_pf < I.n; ++i_pf) {
        /* P(t = i, O) = W->F[i_pf][1] */
        /* P(O)        = W->o          */
        P->PF[I.F[i_pf]].p += c*(W->F[i_pf][1]/W->o);
      }
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < I.m; ++i_ad) {
        /* P(t = i, O) = W->A[i_ad][j] */
        /* P(O)        = W->o          */
        annot_disj_t *AD = &P->AD[I.A[i_ad]];
        for (size_t j = 0; j < AD->n; ++j)
          AD->P[j] += c*(W->A[i_ad][j]/W->o);
      }
    }

    /* Divide probabilistic facts by the number of observations N. */
    for (size_t i_pf = 0; i_pf < I.n; ++i_pf) P->PF[I.F[i_pf]].p /= N;
    /* Divide annotated disjunctions by the number of observations N. */
    for (size_t i_ad = 0; i_ad < I.m; ++i_ad) {
      annot_disj_t *AD = &P->AD[I.A[i_ad]];
      for (size_t j = 0; j < AD->n; ++j)
        AD->P[j] /= N;
    }
  }

  if (!update_program_parameters(P, &I)) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_observations_contents(&O);
  for (size_t i = 0; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_indices_contents(&I);
  return ok;
}

bool learn_lagrange(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat) {
  observations_t O = {0}; /* Observations as a C type. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  indices_t I = {0};
  size_t num_procs = 0;
  bool ok = false;

  if (!init_observations(&O, obs, atoms)) goto cleanup;
  if (!init_indices(&I, P)) goto cleanup;
  Q[0].I_F = I.F; Q[0].n = I.n; Q[0].I_A = I.A; Q[0].m = I.m;
  if (!I.n && !I.m) {
    PyErr_SetString(PyExc_ValueError, "program is not learnable!");
    return false;
  }
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  for (size_t i = 0; i < niters; ++i) {
    /* Compute probabilities. */
    if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, true)) goto cleanup;

    /* Update parameters. */
    for (size_t i_o = 0; i_o < O.n; ++i_o) {
      prob_obs_storage_t *W = &Q[0].P[i_o];
      int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));

      /* Update probabilistic facts. */
      for (size_t i_pf = 0; i_pf < I.n; ++i_pf)
        P->PF[I.F[i_pf]].p += eta*c*(((W->F[i_pf][1] - W->F[i_pf][0])*0.5)/W->o);
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < I.m; ++i_ad) {
        annot_disj_t *AD = &P->AD[I.A[i_ad]];
        double dP = 0.0;
        for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
        for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*c*((W->A[i_ad][j] - dP/AD->n)/W->o);
      }
    }
  }

  if (!update_program_parameters(P, &I)) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_observations_contents(&O);
  for (size_t i = 0; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_indices_contents(&I);
  return ok;
}

bool learn_lagrange_batch(program_t *P, PyArrayObject *obs, size_t niters, double eta,
    size_t batch, bool lstable_sat) {
  observations_t O = {0}; /* Dense representation of observations. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  indices_t I = {0};
  size_t num_procs = 0;
  bool ok = false;

  if (!init_dense_observations(&O, obs, batch)) goto cleanup;
  if (!init_indices(&I, P)) goto cleanup;
  Q[0].I_F = I.F; Q[0].n = I.n; Q[0].I_A = I.A; Q[0].m = I.m;
  Q[0].I_NR = I.NR; Q[0].I_NA = I.NA; Q[0].nr = I.nr; Q[0].na = I.na;
  Q[0].O_NR = I.O_NR; Q[0].O_NA = I.O_NA;
  if (!(I.n || I.m || I.nr || I.na)) {
    PyErr_SetString(PyExc_ValueError, "program is not learnable!");
    return false;
  }
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

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
      for (size_t i_pf = 0; i_pf < I.n; ++i_pf)
        P->PF[I.F[i_pf]].p += eta*(((W->F[i_pf][1] - W->F[i_pf][0])*0.5)/W->o);
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < I.m; ++i_ad) {
        annot_disj_t *AD = &P->AD[I.A[i_ad]];
        double dP = 0.0;
        for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
        for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*((W->A[i_ad][j] - dP/AD->n)/W->o);
      }
      /* Accumulate neural rule derivatives. */
      for (size_t i_nr = 0; i_nr < I.nr; ++i_nr)
        P->NR[I.NR[i_nr]].dw[i_o] = ((W->NR[i_nr][1] - W->NR[i_nr][0])*0.5)/W->o;
      /* Accumulate neural annotated disjunction derivatives. */
      for (size_t i_na = 0; i_na < I.na; ++i_na) {
        neural_annot_disj_t *A = &P->NA[I.NA[i_na]];
        float dP = 0.0;
        for (size_t j = 0; j < A->v; ++j) dP += W->NA[i_na][j];
        for (size_t j = 0; j < A->v; ++j) A->dw[i_o*A->v + j] = (W->NA[i_na][j] - dP/A->v)/W->o;
      }
    }

    /* Backpropagate neural components. */
    for (size_t i_nr = 0; i_nr < I.nr; ++i_nr)
      if (!backward_neural_rule(&P->NR[I.NR[i_nr]], 0, O.n)) goto cleanup;
    for (size_t i_na = 0; i_na < I.na; ++i_na)
      if (!backward_neural_annot_disj(&P->NA[I.NA[i_na]], 0, O.n)) goto cleanup;

    if (!next_dense_observations(&O, obs)) goto cleanup;
  }

  if (!update_program_parameters(P, &I)) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_dense_observations_contents(&O);
  for (size_t i = 0; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_indices_contents(&I);
  return ok;
}

bool learn_neurasp(program_t *P, PyArrayObject *obs, PyArrayObject *obs_counts,
    PyArrayObject *atoms, size_t niters, double eta, bool lstable_sat) {
  observations_t O = {0}; /* Observations as a C type. */
  prob_storage_t Q[NUM_PROCS] = {{0}}; /* Storage for observation probabilities. */
  indices_t I = {0};
  size_t num_procs = 0;
  bool ok = false;

  if (!init_observations(&O, obs, atoms)) goto cleanup;
  if (!init_indices(&I, P)) goto cleanup;
  Q[0].I_F = I.F; Q[0].n = I.n; Q[0].I_A = I.A; Q[0].m = I.m;
  if (!I.n && !I.m) {
    PyErr_SetString(PyExc_ValueError, "program is not learnable!");
    return false;
  }
  if (!(num_procs = init_prob_storage_seq(Q, P, &O))) goto cleanup;

  for (size_t i = 0; i < niters; ++i) {
    /* Compute probabilities. */
    if (!prob_obs_reuse(P, &O, lstable_sat, NULL, Q, true)) goto cleanup;

    /* Update parameters. */
    for (size_t i_o = 0; i_o < O.n; ++i_o) {
      prob_obs_storage_t *W = &Q[0].P[i_o];
      int c = (int) *((int*) PyArray_GETPTR1(obs_counts, i_o));

      /* Update probabilistic facts. */
      for (size_t i_pf = 0; i_pf < I.n; ++i_pf)
        P->PF[I.F[i_pf]].p += eta*c*((W->F[i_pf][1] - W->F[i_pf][0])/W->o);
      /* Update annotated disjunctions. */
      for (size_t i_ad = 0; i_ad < I.m; ++i_ad) {
        annot_disj_t *AD = &P->AD[I.A[i_ad]];
        double dP = 0.0;
        for (size_t j = 0; j < AD->n; ++j) dP += W->A[i_ad][j];
        for (size_t j = 0; j < AD->n; ++j) AD->P[j] += eta*c*((W->A[i_ad][j] - dP)/W->o);
      }
    }
  }

  if (!update_program_parameters(P, &I)) {
    PyErr_SetString(PyExc_AttributeError, "could not update program parameters!");
    goto cleanup;
  }

  ok = true;
cleanup:
  free_observations_contents(&O);
  for (size_t i = 0; i < num_procs; ++i) free_prob_storage_contents(&Q[i], false);
  free_indices_contents(&I);
  return ok;
}

bool update_program_parameters(program_t *P, indices_t *I) {
  for (size_t i = 0; i < I->n; ++i) {
    prob_fact_t *pf = &P->PF[I->F[i]];
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

  for (size_t i = 0; i < I->m; ++i) {
    annot_disj_t *ad = &P->AD[I->A[i]];
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

  return true;
}
