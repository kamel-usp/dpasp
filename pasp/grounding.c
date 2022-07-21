#include <clingo.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int process_error() {
  int code;
  const char *msg;
  if (!(msg = clingo_error_message())) msg = "Unknown error!";
  code = clingo_error_code();
  if (code) printf("Error code %d: %s\n", code, msg);
  return code;
}

typedef struct string {
  char *data;
  size_t n;
} string_t;

static void free_string(string_t *b) {
  if (b->data) {
    free(b->data);
    b->data = NULL;
    b->n = 0;
  }
}

typedef struct body {
  char **subgoals;
  size_t n_subgoals;
  size_t argc;
  char **args;
} body_t;

static body_t* create_body(char **pred, size_t *pred_lens, size_t n_subgoals, size_t argc) {
  size_t i;
  body_t *S;
  S = (body_t*) malloc(sizeof(body_t));
  if (!S) return NULL;
  S->args = NULL;
  S->subgoals = (char**) malloc(n_subgoals*sizeof(char*));
  for (i = 0; i< n_subgoals; ++i) S->subgoals[i] = NULL;
  if (!S->subgoals) goto cleanup;
  for (i = 0; i < n_subgoals; ++i) {
    size_t j = pred_lens[i]+1;
    S->subgoals[i] = (char*) malloc(j*sizeof(char));
    if (!S->subgoals[i]) goto cleanup;
    memcpy(S->subgoals[i], pred[i], j);
  }
  S->args = (char**) malloc(argc*sizeof(char*));
  for (i = 0; i < argc; ++i) S->args[i] = NULL;
  if (!S->args) goto cleanup;
  S->argc = argc; S->n_subgoals = n_subgoals;
  return S;
cleanup:
  if (S->subgoals) for (i = 0; i < n_subgoals; ++i) if(S->subgoals[i]) free(S->subgoals[i]);
  free(S->subgoals); free(S->args); free(S);
  return NULL;
}

static void free_body(body_t *S) {
  size_t i;
  for (i = 0; i < S->n_subgoals; ++i) free(S->subgoals[i]);
  free(S->subgoals);
  for (i = 0; i < S->argc; ++i) free(S->args[i]);
  free(S->args);
  free(S);
}

static bool body_store_arg(body_t *B, char *S, size_t i, size_t begin, size_t end) {
  char *str;
  size_t n = end-begin+1;
  str = (char*) malloc(n*sizeof(char));
  if (!str) return false;
  memcpy(str, S+begin, end-begin);
  str[end] = '\0';
  B->args[i] = str;
  return true;
}

static bool issep(char x) { return x == ',' || x == ';' || x == ' ' || x == ')' || x == '}'; }

static bool body_subgoal_cmp(body_t *G, size_t subgoal, string_t *S, size_t *seen) {
  size_t i = 0, j = 0;
  char *S_s = S->data;
  char *G_s = G->subgoals[subgoal];
  char g, s;

  do {
    g = G_s[i]; s = S_s[j];
    if (g == '%') {
      size_t v, k, l;
      /* Take the variable v surrounded by %s (e.g. in "f(%0%)", v = 0). */
      for (k = i+1; G_s[k] != '%'; ++k);
      G_s[k] = '\0';
      v = atoi(G_s+i+1);
      G_s[k] = '%';
      /* Assume that all variables are increasingly sorted. If v >= seen, we have not yet seen v. */
      if (v >= *seen) {
        /* Save the atom corresponding to v (e.g. when G->pred = "f(%0%)" and S->data "f(alpha)",
         * then we want to save the "alpha" interval as G->args_start and G->args_end. */
        for (l = j; !issep(S_s[l]); ++l);
        body_store_arg(G, S_s, v, j, l);
        /* Update our state. */
        j = l; ++(*seen); i = k+1;
      } else {
        /* If we have seen this variable before, we need only to check if it matches the same one
         * we saved earlier. */
        char *arg = G->args[v];
        for (l = 0; arg[l]; ++l)
          /* If we have no variable match, we already know that the subgoal does not match. */
          if (arg[l] != S_s[j+l]) return false;
        /* Update our state. */
        i = k+1; j += l;
      }
    } else {
      /* If we are not looking at a variable, it suffices to directly match characters. */
      if (g != s) return false;
      ++i; ++j;
    }
  } while (g != '\0' && s != '\0');

  return true;
}

bool print_symbol(clingo_symbol_t sym, string_t *s) {
  char *t;
  size_t n;
  if (!clingo_symbol_to_string_size(sym, &n)) return 0;
  if (s->n < n) {
    if (!(t = (char*) realloc(s->data, sizeof(*s->data)*n))) {
      clingo_set_error(clingo_error_bad_alloc, "Could not allocate memory for symbol's string!");
      return 0;
    }
    s->data = t;
    s->n = n;
  }
  if (!clingo_symbol_to_string(sym, s->data, n)) return 0;
  puts(s->data);
  return 1;
}

static int find(clingo_control_t *C) {
  const clingo_symbolic_atoms_t *atoms;
  clingo_symbolic_atom_iterator_t it, it_end;
  clingo_signature_t u;
  string_t str = {NULL, 0};
  bool ret = false;

  if (!clingo_signature_create("f", 1, true, &u)) goto cleanup;
  if (!clingo_control_symbolic_atoms(C, &atoms)) goto cleanup;
  if (!clingo_symbolic_atoms_begin(atoms, &u, &it)) goto cleanup;
  if (!clingo_symbolic_atoms_end(atoms, &it_end)) goto cleanup;

  while (true) {
    clingo_symbol_t s;
    const clingo_symbol_t *args[] = {NULL, NULL, NULL, NULL};
    size_t n_args, i;
    bool is_end;

    if (!clingo_symbolic_atoms_iterator_is_equal_to(atoms, it, it_end, &is_end)) goto cleanup;
    if (is_end) break;
    if (!clingo_symbolic_atoms_symbol(atoms, it, &s)) goto cleanup;
    if (!print_symbol(s, &str)) goto cleanup;
    if (!clingo_symbol_arguments(s, args, &n_args)) goto cleanup;
    for (i = 0; i < n_args; ++i) if(!print_symbol(*args[i], &str)) goto cleanup;
    if (!clingo_symbolic_atoms_next(atoms, it, &it)) goto cleanup;
  }

  ret = true;
cleanup:
  free_string(&str);
  return ret;
}

static int prepare(const char *P) {
  clingo_control_t *C = NULL;
  clingo_part_t parts[] = {{"base", NULL, 0}};

  if (!clingo_control_new(NULL, 0, NULL, NULL, 20, &C) != 0) goto cleanup;
  if (!clingo_control_add(C, "base", NULL, 0, P)) goto cleanup;
  if (!clingo_control_ground(C, parts, 1, NULL, NULL)) goto cleanup;
  if (!find(C)) goto cleanup;

cleanup:
  if (C) clingo_control_free(C);
  return process_error();
}

static void test_body_cmp() {
  char *c[] = {"f(g(h(%0%),%1%),g(%1%,%0%),%2%,h(%0%))", "p(%3%)", "q(%1%,p(%2%))", "g(p(%2%),f(%3%))"};
  size_t s[] = {38, 6, 13, 16};
  size_t i, seen = 0;
  body_t *G = create_body(c, s, 4, 4);
  string_t S[] = {{"f(g(h(2),3),g(3,2),4,h(2))", 13}, {"p(5)", 4}, {"q(3,p(4))", 9}, {"g(p(4),f(5))", 12}};

  for (i = 0; i < sizeof(S)/sizeof(string_t); ++i)
    printf("-> %d\n", body_subgoal_cmp(G, i, &S[i], &seen));

  free_body(G);
}

int main(void) {
  test_body_cmp();
  return 0;
  prepare("a. b. c. f(1..10). f(g(h(2), 3)).");
}
