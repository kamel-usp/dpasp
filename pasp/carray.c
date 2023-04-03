#include <stdarg.h>
#include <string.h>

#include "carray.h"

#define CARRAY_ARRAY_EXTEND_NP_DECLARE(type) CARRAY_ARRAY_EXTEND_DECLARE(type, type)

ARRAY_IMPL(bool)
/* CARRAY_ARRAY_EXTEND_DECLARE(bool, int); */

ARRAY_IMPL(double)
/* CARRAY_ARRAY_EXTEND_NP_DECLARE(double); */

ARRAY_IMPL(clingo_symbol_t)

/* Finds the next highest power of two integer.
 * See https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float */
#define NEXT_HIGHEST_2_POW_64(x) --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32; ++x;
#define NEXT_HIGHEST_2_POW(x) --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; ++x;

#undef ARRAY_MAGIC_MULTIPLIER
#define ARRAY_MAGIC_MULTIPLIER 2

ARRAY_IMPL(char)

bool array_char_from(array_char_t *a, const char *s) {
  size_t n = strlen(s)+1;
  if (n > a->c) {
    size_t c = n;
    char *z;
    NEXT_HIGHEST_2_POW(c);
    z = (char*) realloc(a->d, c);
    if (!z) return false;
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
    if (!z) return false;
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

ARRAY_IMPL(uint8_t)
