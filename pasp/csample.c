#include "csample.h"

#include "cinf.h"
#include "cutils.h"

typedef struct {
  /* Total choice. */
  total_choice_t theta;
  /* Chunk of the sample matrix. */
  bool *samples;
  /* Size of the chunk. */
  size_t n;
  /* Program. */
  program_t *P;
  /* Atoms to be sampled. */
  clingo_symbol_t *A;
  /* Number of atoms. */
  size_t A_n;
  /* Thread's RNG buffer. */
  unsigned short rng[3];
  /* Process pseudo-PID. */
  size_t pid;
  /* Whether the process has failed. */
  bool fail;
  /* Mutexes and conditional variables. */
  pthread_mutex_t *mu, *wakeup;
  pthread_cond_t *avail;
  /* Busy processes table. */
  bool *busy_procs;
  /* Whether to use the L-stable translation. */
  bool lstable_sat;
} sample_storage_t;

bool atoms2symbols(PyArrayObject *atoms, sample_storage_t S[NUM_PROCS], size_t num_procs) {
  /* Number of elements in atoms. */
  size_t n = PyArray_SIZE(atoms);
  /* Number of bytes in a (possibly wide) character. */
  size_t a = PyArray_DESCR(atoms)->alignment;
  /* Number of bytes in one (possibly wide) string. */
  size_t b = PyArray_ITEMSIZE(atoms);
  clingo_symbol_t *A = NULL;

#define MAX_ATOM_SIZE 256

  if (b/a >= MAX_ATOM_SIZE) {
    PyErr_SetString(PyExc_MemoryError, "atom string is too large!");
    goto cleanup;
  }

  A = (clingo_symbol_t*) malloc(n*sizeof(clingo_symbol_t));
  if (!A) {
    PyErr_SetString(PyExc_MemoryError, "could not allocate memory in atoms2symbols!");
    goto cleanup;
  }

  for (size_t i = 0; i < n; ++i) {
    /* Assume the strings in atoms are unicode, but only take the first characters. */
    size_t j;
    char atom[MAX_ATOM_SIZE];
    char *data = (char*) PyArray_GETPTR1(atoms, i);
    for (size_t c = j = 0; c < b; c += a) atom[j++] = data[c];
    atom[j] = '\0';
    if (!clingo_parse_term(atom, NULL, NULL, 20, &A[i])) goto cleanup;
  }

  for (size_t i = 0; i < num_procs; ++i) {
    S[i].A = A;
    S[i].A_n = n;
  }

  return true;
cleanup:
  free(A);
  return false;
}

bool sample_total_choice(program_t *P, total_choice_t *theta, unsigned short seed[3]) {
  size_t n = P->PF_n;
  size_t m = P->AD_n;

  /* Sample probabilistic facts. */
  for (size_t i = 0; i < n; ++i) bitvec_SET(&theta->pf, i, erand48(seed) <= P->PF[i].p);
  /* Sample annotated disjunctions. */
  for (size_t i = 0; i < m; ++i) {
    double p = P->AD[i].P[0], x = erand48(seed);
    register uint16_t c = 0;
    /* A linear search on the cdf is fine here, since ADs are (usually) small. */
    for (size_t j = 1; (j <= P->AD[i].n) && !c; ++j, p += P->AD[i].P[j-1]) c += j*(x < p);
    theta->theta_ad[i] = c-1;
  }
  return true;
}

void compute_sample(void *args) {
  sample_storage_t *S = (sample_storage_t*) args;
  program_t *P = S->P;
  total_choice_t *theta = &S->theta;

  for (size_t i = 0; i < S->n; ++i) {
    sample_total_choice(P, theta, S->rng);
    clingo_control_t *C = NULL;
    bool gok = false;

    if (P->sem == LSTABLE_SEMANTICS && S->lstable_sat) {
      bool has;
      if (!has_total_model(P, theta, &has)) goto cleanup;
      if (has) P = P->stable;
    }

    if (!prepare_control(&C, P, theta, "0", false, NULL)) goto cleanup;

    size_t m = 0;
    {
      bool ok = false;
      clingo_solve_handle_t *handle;
      clingo_solve_result_bitset_t solve_ret;

      if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle))
        goto count_cleanup;

      for (m = 0; true; ++m) {
        if (!clingo_solve_handle_resume(handle)) goto count_cleanup;
        if (!clingo_solve_handle_get(handle, &solve_ret)) goto count_cleanup;
        if (solve_ret & clingo_solve_result_exhausted) break;
      }

      ok = true;
count_cleanup:
      if (!(clingo_solve_handle_close(handle) && ok)) goto cleanup;
    }
    /* Samples an integer uniformly between 0 and m-1. */
    size_t choice = erand48(S->rng)*m;
    {
      bool ok = false;
      clingo_solve_handle_t *handle;
      clingo_solve_result_bitset_t solve_ret;
      const clingo_model_t *M;

      if (!clingo_control_solve(C, clingo_solve_mode_yield, NULL, 0, NULL, NULL, &handle))
        goto sample_cleanup;

      for (m = 0; true; ++m) {
        if (!clingo_solve_handle_resume(handle)) goto sample_cleanup;
        if (m == choice) {
          if (!clingo_solve_handle_model(handle, &M)) goto sample_cleanup;
          for (size_t j = 0; j < S->A_n; ++j)
            if (!clingo_model_contains(M, S->A[j], S->samples + (i*S->A_n+j))) goto sample_cleanup;
          break;
        }
        if (!clingo_solve_handle_get(handle, &solve_ret)) goto sample_cleanup;
        if (solve_ret & clingo_solve_result_exhausted) break;
      }

      ok = true;
sample_cleanup:
      if (!(clingo_solve_handle_close(handle) && ok)) goto cleanup;
    }

    gok = true;
cleanup:
    if (C) clingo_control_free(C);
    if (!gok) { S->fail = true; break; }
  }
}

#define min(x, y) ((x) > (y) ? (y) : (x))
#define max(x, y) ((x) > (y) ? (x) : (y))

bool naive_sample(program_t *P, size_t n, PyArrayObject *atoms, bool lstable_sat, PyObject **ret) {
  import_array();
  size_t total_choice_n = get_num_facts(P);
  /* Heuristic for choosing the number of processes to use. Roughly 100 samples per process. */
  size_t num_procs = max(min(n / 100, NUM_PROCS), 1);
  bool ok = false;
  threadpool pool = thpool_init(num_procs);
  sample_storage_t S[NUM_PROCS] = {0};
  bool *samples = NULL;
  size_t m = (size_t) PyArray_SIZE(atoms);

  /* Variable samples is a matrix of dimension n by m in contiguous array format. */
  samples = (bool*) malloc(n*m*sizeof(bool));
  if (!samples) goto cleanup;

  /* Initialize storages. */ {
    size_t d = n / num_procs;
    size_t r = n % num_procs;
    size_t t = 0;
    for (size_t i = 0; i < num_procs; ++i) {
      S[i].pid = i;
      S[i].lstable_sat = lstable_sat;
      S[i].P = P;
      S[i].fail = false;
      if (!init_total_choice(&S[i].theta, total_choice_n, P)) goto cleanup;
      /* Split samples into num_procs approximately equally sized chunks for parallelism. */
      S[i].n = d + (i < r);
      S[i].samples = samples + t;
      S[i].rng[0] = rand(); S[i].rng[1] = rand(); S[i].rng[2] = rand();
      t += m*S[i].n;
    }
  }
  if (!atoms2symbols(atoms, S, num_procs)) goto cleanup;

  for (size_t i = 0; i < num_procs; ++i)
    if (thpool_add_work(pool, compute_sample, &S[i])) goto cleanup;
  thpool_wait(pool);
  for (size_t i = 0; i < num_procs; ++i) if (S[i].fail) goto cleanup;

  npy_intp dims[2] = {n, m};
  *ret = PyArray_SimpleNewFromData(2, dims, NPY_BOOL, samples);
  if (!*ret) goto cleanup;
  PyArray_ENABLEFLAGS((PyArrayObject*) *ret, NPY_ARRAY_OWNDATA);

  ok = true;
cleanup:
  if (clingo_error_code() != clingo_error_success) raise_clingo_error(NULL);
  thpool_destroy(pool);
  for (size_t i = 0; i < num_procs; ++i) free_total_choice_contents(&S[i].theta);
  free(S[0].A);
  if (!ok) free(samples);
  return ok;
}
