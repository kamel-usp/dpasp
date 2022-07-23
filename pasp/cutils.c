#ifndef PASP_CUTILS
#define PASP_CUTILS

#include <clingo.h>
#include <stdlib.h>

typedef struct string {
  char *s;
  size_t n;
} string_t;

static bool string_from_symbol(clingo_symbol_t sym, string_t *buf) {
  bool r = true;
  char *s;
  size_t n;

  if (!clingo_symbol_to_string_size(sym, &n)) goto error;
  if (buf->n < n) {
    if (!(s = (char*) realloc(buf->s, n*sizeof(char)))) {
      clingo_set_error(clingo_error_bad_alloc, "Could not allocate memory for symbol!");
      goto error;
    }
    buf->s = s;
    buf->n = n;
  }
  if (!clingo_symbol_to_string(sym, buf->s, n)) goto error;
  goto out;
error:
  r = false;
out:
  return r;
}

static void string_free(string_t *s) {
  if (!s->s) return;
  free(s->s);
  s->s = NULL, s->n = 0;
}

int main() { return 0; }

#endif
