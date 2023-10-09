#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "ctree.h"

#include <stdio.h>
#include <string.h>

ctree_t* ctree_create(void) {
  ctree_t *T = (ctree_t*) malloc(sizeof(ctree_t));
  if (!T) return NULL;
  ctree_init(T);
  return T;
}
static inline bool _ctree_init(ctree_t *T, size_t n) {
  T->ch = (ctree_t*) malloc(n*sizeof(ctree_t));
  if (!T->ch) {
    PyErr_SetString(PyExc_MemoryError, "could not allocate enough memory for ctree!");
    return false;
  }
  for (size_t i = 0; i < n; ++i) { T->ch[i].ch = NULL; T->ch[i].n = 0; T->pr = 0.0; }
  T->n = n;
  return true;
}
void ctree_init(ctree_t *T) {
  T->ch = NULL; T->n = 0; T->pr = 0.0;
}

void ctree_free(ctree_t *T) { ctree_free_contents(T); free(T); }
void ctree_free_contents(ctree_t *T) {
  if (!T->ch) return;
  for (size_t i = 0; i < T->n; ++i) ctree_free_contents(&T->ch[i]);
  free(T->ch);
  T->ch = NULL;
}

#define CTREE_IS_EMPTY(T) (!(((T)->ch) || ((T)->n)))
ctree_t* _ctree_add(ctree_t *T, const clingo_model_t *M, program_t *P, size_t i_pf, size_t i_ad,
    size_t i_nr, size_t i_na, size_t neural_offset, double p) {
  /* Arguments t_nr and t_na are the total number of NR and NA (after grounding and accounting for
   * outcomes) respectively. Argument neural_offset is the index of the current test instance.
   * Arguments i_nr and i_na are the indices of neural rules and neural annotated disjunctions
   * respectively.*/
  if (i_pf < P->PF_n) { /* Probabilistic facts. */
    /* If node is empty, then populate. */
    if (CTREE_IS_EMPTY(T)) if (!_ctree_init(T, 2)) return NULL;
    /* Check which path to take according to the presence of the probabilistic fact. */
    bool c;
    if (!clingo_model_contains(M, P->PF[i_pf].cl_f, &c)) return NULL;
    return _ctree_add(T->ch+c, M, P, i_pf+1, i_ad, i_nr, i_na, neural_offset,
        p*(c*P->PF[i_pf].p + (!c)*(1-P->PF[i_pf].p)));
  } else if (i_ad < P->AD_n) { /* Annotated Disjunctions. */
    /* If node is empty, then populate. */
    if (CTREE_IS_EMPTY(T)) if (!_ctree_init(T, P->AD[i_ad].n)) return NULL;
    /* Check which path to take according to the presence of the probabilistic fact. */
    bool c;
    for (size_t i = 0; i < P->AD[i_ad].n; ++i) {
      if (!clingo_model_contains(M, P->AD[i_ad].cl_F[i], &c)) return NULL;
      if (c) return _ctree_add(T->ch+i, M, P, i_pf, i_ad+1, i_nr, i_na, neural_offset,
          p*P->AD[i_ad].P[i]);
    }
    /* We should never get down here! Every model has a total choice and every total choice assigns a
     * choice to every probabilistic component. */
    return NULL;
  } else if (i_nr < P->NR_n) { /* Neural rules. */
    if (CTREE_IS_EMPTY(T)) if (!_ctree_init(T, 2)) return NULL;
    /* TODO: when implementing learning, m has to be P->batch. */
    size_t m = P->m_test;
    float *Pr = P->NR[i_nr].P+neural_offset*P->NR[i_nr].o, pr;
    bool c;
    for (size_t g = 0; g < P->NR[i_nr].n; ++g)
      for (size_t o = 0; o < P->NR[i_nr].o; ++o) {
        pr = Pr[g*P->NR[i_nr].o*m+o];
        if (!clingo_model_contains(M, P->NR[i_nr].H[g*P->NR[i_nr].o+o], &c)) return NULL;
        T = T->ch+c;
        if (CTREE_IS_EMPTY(T)) if (!_ctree_init(T, 2)) return NULL;
        p *= c*pr + (!c)*(1-pr);
      }
    return _ctree_add(T, M, P, i_pf, i_ad, i_nr+1, i_na, neural_offset, p);
  } else if (i_na < P->NA_n) { /* Neural annotated disjunctions. */
    if (CTREE_IS_EMPTY(T)) if (!_ctree_init(T, P->NA[i_na].v)) return NULL;
    /* TODO: when implementing learning, m has to be P->batch. */
    size_t m = P->m_test;
    float *Pr = P->NA[i_na].P+neural_offset*P->NA[i_na].v*P->NA[i_na].o, pr;
    bool c;
    for (size_t g = 0; g < P->NA[i_na].n; ++g)
      for (size_t o = 0; o < P->NA[i_na].o; ++o) {
        size_t offset = g*P->NA[i_na].v*P->NA[i_na].o+o*P->NA[i_na].v;
        for (size_t v = 0; v < P->NA[i_na].v; ++v) {
          pr = Pr[g*P->NA[i_na].o*m+o+v];
          if (!clingo_model_contains(M, P->NA[i_na].H[offset+v], &c)) return NULL;
          if (!c) continue;
          T = T->ch+v;
          if (CTREE_IS_EMPTY(T)) if (!_ctree_init(T, P->NA[i_na].v)) return NULL;
          p *= pr;
        }
      }
    return _ctree_add(T, M, P, i_pf, i_ad, i_nr, i_na+1, neural_offset, p);
  }
  /* Leaf. */
  T->ch = NULL; ++T->n; T->pr = p;
  return T;
}

ctree_t* ctree_add(ctree_t *T, const clingo_model_t *M, program_t *P, size_t neural_offset) {
  return _ctree_add(T, M, P, 0, 0, 0, 0, neural_offset, 1.0);
}

bool _ctree_dot(ctree_t *T, program_t *P, size_t i_pf, size_t i_ad, char *buffer, size_t *offset) {
  static size_t id = 0;
  size_t this = id;
  if (i_pf < P->PF_n) {
    *offset += sprintf(buffer+*offset, "  %lu [label=\"%s\"]\n", id, P->PF[i_pf].f);
    if (T->ch[0].ch || T->ch[0].n) {
      *offset += sprintf(buffer+*offset, "  %lu -> %lu [label=\"0\"]\n", this, id+1); ++id;
      if (!_ctree_dot(&T->ch[0], P, i_pf+1, i_ad, buffer, offset)) return false;
    }
    if (T->ch[1].ch || T->ch[1].n) {
      *offset += sprintf(buffer+*offset, "  %lu -> %lu [label=\"1\"]\n", this, id+1); ++id;
      if (!_ctree_dot(&T->ch[1], P, i_pf+1, i_ad, buffer, offset)) return false;
    }
    return true;
  } else if (i_ad < P->AD_n) {
    *offset += sprintf(buffer+*offset, "  %lu [label=\"%s\"]\n", this, P->AD[i_ad].F[0]);
    for (size_t i = 0; i < P->AD[i_ad].n; ++i)
      if (T->ch[i].ch || T->ch[i].n) {
        *offset += sprintf(buffer+*offset, "  %lu -> %lu [label=\"%lu\"]\n", this, id+1, i+1); ++id;
        if (!_ctree_dot(&T->ch[i], P, i_pf, i_ad+1, buffer, offset)) return false;
      }
    return true;
  }
  *offset += sprintf(buffer+*offset, "  %lu [label=\"%lu\"]\n", this, T->n);
  return true;
}
bool ctree_dot(ctree_t *T, program_t *P, char *buffer) {
  size_t offset = sprintf(buffer, "strict digraph {\n");
  if (!_ctree_dot(T, P, 0, 0, buffer, &offset)) return false;
  strcat(buffer, "}\n");
  return true;
}
