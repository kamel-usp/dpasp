#ifndef _PASP_CARRAY
#define _PASP_CARRAY

#include <stdlib.h>
#include <stdbool.h>
#include <clingo.h>

#define CARRAY_ARRAY_TYPE_DECLARE(type) \
  typedef struct array_##type { \
    type *d; /* data */ \
    size_t c; /* capacity */ \
    size_t n; /* size */ \
  } array_##type##_t

CARRAY_ARRAY_TYPE_DECLARE(bool);
CARRAY_ARRAY_TYPE_DECLARE(double);
CARRAY_ARRAY_TYPE_DECLARE(char);
CARRAY_ARRAY_TYPE_DECLARE(clingo_symbol_t);

#define CARRAY_ARRAY_HEADER(type, ret_type, func) \
ret_type array_##type##_##func(array_##type##_t *a)
#define CARRAY_ARRAY_HEADER_PONE(type, ret_type, func) \
ret_type array_##type##_##func(array_##type##_t *a, type o)

CARRAY_ARRAY_HEADER(bool, bool, init);
CARRAY_ARRAY_HEADER(bool, void, free_contents);
CARRAY_ARRAY_HEADER(bool, void, free);
CARRAY_ARRAY_HEADER_PONE(bool, bool, append);

CARRAY_ARRAY_HEADER(double, bool, init);
CARRAY_ARRAY_HEADER(double, void, free_contents);
CARRAY_ARRAY_HEADER(double, void, free);
CARRAY_ARRAY_HEADER_PONE(double, bool, append);

CARRAY_ARRAY_HEADER(clingo_symbol_t, bool, init);
CARRAY_ARRAY_HEADER(clingo_symbol_t, void, free_contents);
CARRAY_ARRAY_HEADER(clingo_symbol_t, void, free);
CARRAY_ARRAY_HEADER_PONE(clingo_symbol_t, bool, append);

CARRAY_ARRAY_HEADER(char, bool, init);
CARRAY_ARRAY_HEADER(char, void, free_contents);
CARRAY_ARRAY_HEADER(char, void, free);
CARRAY_ARRAY_HEADER_PONE(char, bool, append);

bool array_char_from(array_char_t *a, const char *s);
bool array_char_writeln(array_char_t *a, char *s, size_t n);

#endif
