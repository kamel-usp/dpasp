#ifndef Py_CARRAYMODULE_H
#define Py_CARRAYMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdbool.h>

#define CARRAY_ARRAY_TYPE_DECLARE(type) \
  typedef struct array_##type { \
    type *d; /* data */ \
    size_t c; /* capacity */ \
    size_t n; /* size */ \
  } array_##type##_t;

CARRAY_ARRAY_TYPE_DECLARE(bool);
CARRAY_ARRAY_TYPE_DECLARE(double);

#define PyCarray_array_double_init_NUM 0
#define PyCarray_array_double_init_RETURN bool
#define PyCarray_array_double_init_PROTO (array_double_t *a)

#define PyCarray_array_bool_init_NUM 1
#define PyCarray_array_bool_init_RETURN bool
#define PyCarray_array_bool_init_PROTO (array_bool_t *a)

#define PyCarray_array_double_free_contents_NUM 2
#define PyCarray_array_double_free_contents_RETURN void
#define PyCarray_array_double_free_contents_PROTO (array_double_t *a)

#define PyCarray_array_bool_free_contents_NUM 3
#define PyCarray_array_bool_free_contents_RETURN void
#define PyCarray_array_bool_free_contents_PROTO (array_bool_t *a)

#define PyCarray_array_double_free_NUM 4
#define PyCarray_array_double_free_RETURN void
#define PyCarray_array_double_free_PROTO (array_double_t *a)

#define PyCarray_array_bool_free_NUM 5
#define PyCarray_array_bool_free_RETURN void
#define PyCarray_array_bool_free_PROTO (array_bool_t *a)

#define PyCarray_array_double_append_NUM 6
#define PyCarray_array_double_append_RETURN bool
#define PyCarray_array_double_append_PROTO (array_double_t *a, double o)

#define PyCarray_array_bool_append_NUM 7
#define PyCarray_array_bool_append_RETURN bool
#define PyCarray_array_bool_append_PROTO (array_bool_t *a, bool o)

#define PyCarray_API_pointers 8

#ifdef CARRAY_MODULE

static PyCarray_array_bool_init_RETURN array_bool_init PyCarray_array_bool_init_PROTO;
static PyCarray_array_bool_free_RETURN array_bool_free PyCarray_array_bool_free_PROTO;
static PyCarray_array_bool_free_contents_RETURN array_bool_free_contents PyCarray_array_bool_free_contents_PROTO;
static PyCarray_array_bool_append_RETURN array_bool_append PyCarray_array_bool_append_PROTO;

static PyCarray_array_double_init_RETURN array_double_init PyCarray_array_double_init_PROTO;
static PyCarray_array_double_free_RETURN array_double_free PyCarray_array_double_free_PROTO;
static PyCarray_array_double_free_contents_RETURN array_double_free_contents PyCarray_array_double_free_contents_PROTO;
static PyCarray_array_double_append_RETURN array_double_append PyCarray_array_double_append_PROTO;

#else

static void** PyCarray_API;

#define array_bool_init \
  (*(PyCarray_array_bool_init_RETURN (*)PyCarray_array_bool_init_PROTO) PyCarray_API[PyCarray_array_bool_init_NUM])
#define array_bool_free_contents \
  (*(PyCarray_array_bool_free_contents_RETURN (*)PyCarray_array_bool_free_contents_PROTO) PyCarray_API[PyCarray_array_bool_free_contents_NUM])
#define array_bool_free \
  (*(PyCarray_array_bool_free_RETURN (*)PyCarray_array_bool_free_PROTO) PyCarray_API[PyCarray_array_bool_free_NUM])
#define array_bool_append \
  (*(PyCarray_array_bool_append_RETURN (*)PyCarray_array_bool_append_PROTO) PyCarray_API[PyCarray_array_bool_append_NUM])
#define array_double_init \
  (*(PyCarray_array_double_init_RETURN (*)PyCarray_array_double_init_PROTO) PyCarray_API[PyCarray_array_double_init_NUM])
#define array_double_free_contents \
  (*(PyCarray_array_double_free_contents_RETURN (*)PyCarray_array_double_free_contents_PROTO) PyCarray_API[PyCarray_array_double_free_contents_NUM])
#define array_double_free \
  (*(PyCarray_array_double_free_RETURN (*)PyCarray_array_double_free_PROTO) PyCarray_API[PyCarray_array_double_free_NUM])
#define array_double_append \
  (*(PyCarray_array_double_append_RETURN (*)PyCarray_array_double_append_PROTO) PyCarray_API[PyCarray_array_double_append_NUM])

static int import_carray(void) {
  PyCarray_API = (void**) PyCapsule_Import("carray._C_API", 0);
  return (PyCarray_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
