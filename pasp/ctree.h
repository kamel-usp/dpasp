#ifndef _PASP_CTREE
#define _PASP_CTREE

#include <stdlib.h>
#include <stdbool.h>
#include <clingo.h>

#include "cprogram.h"

typedef struct ctree ctree_t;
/** Total choice tree for counting models. */
struct ctree {
  /** Children. */
  ctree_t *ch;
  /** If leaf node (i.e. ch == NULL), then n is number of models of this total choice. Else, it is
   * the number of children of this node. */
  size_t n;
  /** Probability up to this node. */
  double pr;
};

/** Creates a new ctree_t. */
ctree_t* ctree_create(void);
/** Initializes a ctree_t. */
void ctree_init(ctree_t *T);
#define CTREE_INIT {0}

/** Frees the memory of a ctree_t. */
void ctree_free(ctree_t *T);
/** Frees the memory contents of a ctree_t. */
void ctree_free_contents(ctree_t *T);

/** Add model to ctree_t T. */
ctree_t* ctree_add(ctree_t *T, const clingo_model_t *M, program_t *P);

/** Write T as a dot file in buffer. */
bool ctree_dot(ctree_t *T, program_t *P, char *buffer);

#endif
