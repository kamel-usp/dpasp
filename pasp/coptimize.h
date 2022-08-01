#ifndef Py_COPTIMIZEMODULE_H
#define Py_COPTIMIZEMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef bool
#define REDEF_BOOL
#define bool char
#endif

#define PyCoptimize_bfca_NUM 0
#define PyCoptimize_bfca_RETURN double
#define PyCoptimize_bfca_PROTO (double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, \
    double *L, double *U, size_t n_a, size_t n_b, size_t m, int maxmin, size_t tries, bool smp)

#define PyCoptimize_bf_NUM 1
#define PyCoptimize_bf_RETURN void
#define PyCoptimize_bf_PROTO (double *X, bool *S_a, bool *S_b, double *C_a, double *C_b, \
    double *L, double *U, size_t n_a, size_t n_b, size_t m, double *low, double *up, bool smp)

#define PyCoptimize_bf_minmax_NUM 2
#define PyCoptimize_bf_minmax_RETURN void
#define PyCoptimize_bf_minmax_PROTO (double *X, bool *S_a, bool *S_b, bool *S_c, bool* S_d, \
    double *C_a, double *C_b, double *C_c, double *C_d, double *L, double *U, size_t n_a, \
    size_t n_b, size_t n_c, size_t n_d, size_t m, double *low, double *up)

#define PyCoptimize_API_pointers 3

#ifdef COPTIMIZE_MODULE

static PyCoptimize_bfca_RETURN bfca PyCoptimize_bfca_PROTO;
static PyCoptimize_bf_RETURN bf PyCoptimize_bf_PROTO;
static PyCoptimize_bf_minmax_RETURN bf_minmax PyCoptimize_bf_minmax_PROTO;

#else

static void** PyCoptimize_API;

#define bfca \
  (*(PyCoptimize_bfca_RETURN (*)PyCoptimize_bfca_PROTO) PyCoptimize_API[PyCoptimize_bfca_NUM])
#define bf \
  (*(PyCoptimize_bf_RETURN (*)PyCoptimize_bf_PROTO) PyCoptimize_API[PyCoptimize_bf_NUM])
#define bf_minmax \
  (*(PyCoptimize_bf_minmax_RETURN (*)PyCoptimize_bf_minmax_PROTO) PyCoptimize_API[PyCoptimize_bf_minmax_NUM])

static int import_coptimize(void) {
  PyCoptimize_API = (void**) PyCapsule_Import("coptimize._C_API", 0);
  return (PyCoptimize_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#ifdef REDEF_BOOL
#undef bool
#undef REDEF_BOOL
#endif

#endif
