#ifndef _PASP_CARRAY
#define _PASP_CARRAY

#include <stdlib.h>
#include <stdbool.h>
#include <clingo.h>

#define ARRAY_MAGIC_MULTIPLIER 1.5
#define ARRAY_MAGIC_INIT_CAP   16

#define CARRAY_ARRAY_TYPE_DECLARE(type) \
  typedef struct array_##type { \
    type *d; /* data */ \
    size_t c; /* capacity */ \
    size_t n; /* size */ \
  } array_##type##_t

#define CARRAY_ARRAY_HEADER(type, ret_type, func) \
ret_type array_##type##_##func(array_##type##_t *a)
#define CARRAY_ARRAY_HEADER_PONE(type, ret_type, func) \
ret_type array_##type##_##func(array_##type##_t *a, type o)
#define CARRAY_ARRAY_HEADER_ARG(type, ret_type, arg_type, func) \
ret_type array_##type##_##func(array_##type##_t *a, arg_type o)

#define CARRAY_ARRAY_INIT_DECLARE(type) \
bool array_##type##_init(array_##type##_t *a) { \
  a->d = (type*) malloc(ARRAY_MAGIC_INIT_CAP*sizeof(type)); \
  if (!a->d) { \
    a->c = 0; \
    return false; \
  } \
  a->c = ARRAY_MAGIC_INIT_CAP; \
  a->n = 0; \
  return true; \
}

#define CARRAY_ARRAY_INITN_DECLARE(type) \
bool array_##type##_initn(array_##type##_t *a, size_t c) { \
  a->d = (type*) malloc(c*sizeof(type)); \
  if (!a->d) { \
    a->c = 0; \
    return false; \
  } \
  a->c = c; \
  a->n = 0; \
  return true; \
}

#define CARRAY_ARRAY_FREE_CONTENTS_DECLARE(type) \
void array_##type##_free_contents(array_##type##_t *a) { \
  if (a->d) { free(a->d); a->d = NULL; a->c = 0; a->n = 0; } \
}

#define CARRAY_ARRAY_FREE_DECLARE(type) \
void array_##type##_free(array_##type##_t *a) { array_##type##_free_contents(a); free(a); }

#define CARRAY_ARRAY_GROW_DECLARE(type) \
bool array_##type##_grow(array_##type##_t *a) { \
  size_t nc = ARRAY_MAGIC_MULTIPLIER*a->c; \
  size_t nb = sizeof(type)*nc; \
  type *d = (type*) realloc(a->d, nb); \
  if (!d) return false; \
  a->d = d; \
  a->c = nc; \
  return true; \
}

#define CARRAY_ARRAY_APPEND_DECLARE(type) \
bool array_##type##_append(array_##type##_t *a, type o) { \
  if (a->n + 1 > a->c) { \
    if (!array_##type##_grow(a)) return false; \
    a->d[a->n++] = o; \
  } else a->d[a->n++] = o; \
  return true; \
}

#define CARRAY_ARRAY_EXTEND_DECLARE(type, promoted) \
bool array_##type##_extend(array_##type##_t *a, size_t argc, ...) { \
  va_list args; \
  if ((a->n + argc > a->c)) if (!array_##type##_grow(a)) return false; \
  va_start(args, argc); \
  while (argc--) a->d[a->n++] = va_arg(args, promoted); \
  return true; \
}

#define CARRAY_ARRAY_CLEAR_DECLARE(type) void array_##type##_clear(array_##type##_t *a) { a->n = 0; }

#define ARRAY_IMPL(t) \
  CARRAY_ARRAY_INIT_DECLARE(t) \
  CARRAY_ARRAY_INITN_DECLARE(t) \
  CARRAY_ARRAY_FREE_CONTENTS_DECLARE(t) \
  CARRAY_ARRAY_FREE_DECLARE(t) \
  CARRAY_ARRAY_GROW_DECLARE(t) \
  CARRAY_ARRAY_APPEND_DECLARE(t) \
  CARRAY_ARRAY_CLEAR_DECLARE(t)

#define ARRAY_DECL(t) \
  CARRAY_ARRAY_TYPE_DECLARE(t); \
  CARRAY_ARRAY_HEADER(t, bool, init); \
  CARRAY_ARRAY_HEADER(t, void, free_contents); \
  CARRAY_ARRAY_HEADER(t, void, free); \
  CARRAY_ARRAY_HEADER_PONE(t, bool, append); \
  CARRAY_ARRAY_HEADER_ARG(t, bool, size_t, initn); \
  CARRAY_ARRAY_HEADER(t, void, clear);

ARRAY_DECL(bool)
typedef array__Bool_t array_bool_t;
#define array_bool_init array__Bool_init
#define array_bool_free_contents array__Bool_free_contents
#define array_bool_free array__Bool_free
#define array_bool_append array__Bool_append
#define array_bool_initn array__Bool_initn
#define array_bool_clear array__Bool_clear

ARRAY_DECL(double)
ARRAY_DECL(char)
ARRAY_DECL(clingo_symbol_t)
typedef array_clingo_symbol_t_t array_symbol_t;
ARRAY_DECL(uint8_t)
typedef array_uint8_t_t array_uint8_t;

bool array_char_from(array_char_t *a, const char *s);
bool array_char_writeln(array_char_t *a, char *s, size_t n);

#endif
