#include <stdarg.h>
#include <string.h>

#include "carray.h"

#define ARRAY_MAGIC_MULTIPLIER 1.5
#define ARRAY_MAGIC_INIT_CAP   16

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
#define CARRAY_ARRAY_EXTEND_NP_DECLARE(type) CARRAY_ARRAY_EXTEND_DECLARE(type, type)

CARRAY_ARRAY_INIT_DECLARE(bool)
CARRAY_ARRAY_FREE_CONTENTS_DECLARE(bool)
CARRAY_ARRAY_FREE_DECLARE(bool)
CARRAY_ARRAY_GROW_DECLARE(bool)
CARRAY_ARRAY_APPEND_DECLARE(bool)
/* CARRAY_ARRAY_EXTEND_DECLARE(bool, int); */

CARRAY_ARRAY_INIT_DECLARE(double)
CARRAY_ARRAY_FREE_CONTENTS_DECLARE(double)
CARRAY_ARRAY_FREE_DECLARE(double)
CARRAY_ARRAY_GROW_DECLARE(double)
CARRAY_ARRAY_APPEND_DECLARE(double)
/* CARRAY_ARRAY_EXTEND_NP_DECLARE(double); */

CARRAY_ARRAY_INIT_DECLARE(clingo_symbol_t)
CARRAY_ARRAY_FREE_CONTENTS_DECLARE(clingo_symbol_t)
CARRAY_ARRAY_FREE_DECLARE(clingo_symbol_t)
CARRAY_ARRAY_GROW_DECLARE(clingo_symbol_t)
CARRAY_ARRAY_APPEND_DECLARE(clingo_symbol_t)

/* Finds the next highest power of two integer.
 * See https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float */
#define NEXT_HIGHEST_2_POW_64(x) --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32; ++x;
#define NEXT_HIGHEST_2_POW(x) --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; ++x;

#undef ARRAY_MAGIC_MULTIPLIER
#define ARRAY_MAGIC_MULTIPLIER 2
CARRAY_ARRAY_INIT_DECLARE(char)
CARRAY_ARRAY_FREE_CONTENTS_DECLARE(char)
CARRAY_ARRAY_FREE_DECLARE(char)
CARRAY_ARRAY_GROW_DECLARE(char)
CARRAY_ARRAY_APPEND_DECLARE(char)
bool array_char_from(array_char_t *a, const char *s) {
  size_t n = strlen(s)+1;
  if (n > a->c) {
    size_t c = n;
    char *z;
    NEXT_HIGHEST_2_POW(c);
    z = (char*) realloc(a->d, c);
    if (!z) {
      a->d = NULL;
      return false;
    }
    a->d = z;
    a->c = c;
  }
  memcpy(a->d, s, n);
  a->n = n;
  return true;
}
bool array_char_writeln(array_char_t *a, char *s, size_t n) {
  if (!n) n = strlen(s)+1;
  if ((a->n + n) > a->c) {
    size_t c = a->n + n;
    char *z;
    NEXT_HIGHEST_2_POW(c);
    z = (char*) realloc(a->d, c);
    if (!z) {
      a->d = NULL;
      return false;
    }
    a->d = z;
    a->c = c;
  }
  if (a->n) a->d[a->n-1] = '\n';
  memcpy(a->d + a->n, s, n);
  a->n += n - 1;
  return true;
}
#undef ARRAY_MAGIC_MULTIPLIER
#define ARRAY_MAGIC_MULTIPLIER 1.5
