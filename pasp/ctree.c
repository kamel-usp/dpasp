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
}

#define CTREE_IS_EMPTY(T) (!(((T)->ch) || ((T)->n)))
ctree_t* _ctree_add(ctree_t *T, const clingo_model_t *M, program_t *P, size_t i_pf, size_t i_ad,
    double p) {
  if (i_pf < P->PF_n) {
    /* If node is empty, then populate. */
    if (CTREE_IS_EMPTY(T)) if (!_ctree_init(T, 2)) return NULL;
    /* Check which path to take according to the presence of the probabilistic fact. */
    bool c;
    if (!clingo_model_contains(M, P->PF[i_pf].cl_f, &c)) return NULL;
    return _ctree_add(T->ch+c, M, P, i_pf+1, i_ad, p*(c*P->PF[i_pf].p + (!c)*(1-P->PF[i_pf].p)));
  } else if (i_ad < P->AD_n) {
    /* If node is empty, then populate. */
    if (CTREE_IS_EMPTY(T)) if (!_ctree_init(T, P->AD[i_ad].n)) return NULL;
    /* Check which path to take according to the presence of the probabilistic fact. */
    bool c;
    for (size_t i = 0; i < P->AD[i_ad].n; ++i) {
      if (!clingo_model_contains(M, P->AD[i_ad].cl_F[i], &c)) return NULL;
      if (c) return _ctree_add(T->ch+i, M, P, i_pf, i_ad+1, p*P->AD[i_ad].P[i]);
    }
    /* We should never get down here! Every model has a total choice and every total choice assigns a
     * choice to every probabilistic component. */
    return NULL;
  }
  /* Leaf. */
  ++T->n; T->pr = p;
  return T;
}

ctree_t* ctree_add(ctree_t *T, const clingo_model_t *M, program_t *P) {
  return _ctree_add(T, M, P, 0, 0, 1.0);
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
