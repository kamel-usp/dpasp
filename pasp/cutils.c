#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "cutils.h"

const clingo_part_t GROUND_DEFAULT_PARTS[] = {{"base", NULL, 0}};
const char* const CONTROL_DEFAULT_ARGS[] = {"0"};
const size_t CONTROL_DEFAULT_NARGS = 1;

bool char_from_symbol(clingo_symbol_t sym, char *s, size_t n) {
  size_t k;
  if (!clingo_symbol_to_string_size(sym, &k)) return false;
  if (n < k) return false;
  if (!clingo_symbol_to_string(sym, s, k)) return false;
  return true;
}

bool string_from_symbol(clingo_symbol_t sym, string_t *buf) {
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

void string_free(string_t *s) {
  if (!s->s) return;
  free(s->s);
  s->s = NULL, s->n = 0;
}

typedef struct model_buffer {
  clingo_symbol_t *symbols;
  size_t           symbols_n;
  char            *string;
  size_t           string_n;
} model_buffer_t;

void free_model_buffer(model_buffer_t *buf) {
  if (buf->symbols) {
    free(buf->symbols);
    buf->symbols   = NULL;
    buf->symbols_n = 0;
  }
  if (buf->string) {
    free(buf->string);
    buf->string   = NULL;
    buf->string_n = 0;
  }
}

bool print_symbol(clingo_symbol_t symbol, model_buffer_t *buf) {
  bool ret = true;
  char *string;
  size_t n;
  // determine size of the string representation of the next symbol in the model
  if (!clingo_symbol_to_string_size(symbol, &n)) { goto error; }
  if (buf->string_n < n) {
    // allocate required memory to hold the symbol's string
    if (!(string = (char*)realloc(buf->string, sizeof(*buf->string) * n))) {
      clingo_set_error(clingo_error_bad_alloc, "could not allocate memory for symbol's string");
      goto error;
    }
    buf->string   = string;
    buf->string_n = n;
  }
  // retrieve the symbol's string
  if (!clingo_symbol_to_string(symbol, buf->string, n)) { goto error; }
  printf("%s", buf->string);
  goto out;
error:
  ret = false;
out:
  return ret;
}

bool print_model(const clingo_model_t *model, model_buffer_t *buf, const char *label, clingo_show_type_bitset_t show) {
  bool ret = true;
  clingo_symbol_t *symbols;
  size_t n;
  const clingo_symbol_t *it, *ie;
  // determine the number of (shown) symbols in the model
  if (!clingo_model_symbols_size(model, show, &n)) { goto error; }
  // allocate required memory to hold all the symbols
  if (buf->symbols_n < n) {
    if (!(symbols = (clingo_symbol_t*)malloc(sizeof(*buf->symbols) * n))) {
      clingo_set_error(clingo_error_bad_alloc, "could not allocate memory for atoms");
      goto error;
    }
    buf->symbols   = symbols;
    buf->symbols_n = n;
  }
  // retrieve the symbols in the model
  if (!clingo_model_symbols(model, show, buf->symbols, n)) { goto error; }
  printf("%s:", label);
  for (it = buf->symbols, ie = buf->symbols + n; it != ie; ++it) {
    printf(" ");
    if (!print_symbol(*it, buf)) { goto error; }
  }
  printf("\n");
  goto out;
error:
  ret = false;
out:
  return ret;
}

bool print_solution(const clingo_model_t *model) {
  bool ret = true;
  uint64_t number;
  clingo_model_type_t type;
  const char *type_string = "";
  model_buffer_t data = {NULL, 0, NULL, 0};
  // get model type
  if (!clingo_model_type(model, &type)) { goto error; }
  switch ((enum clingo_model_type_e)type) {
    case clingo_model_type_stable_model:          { type_string = "Stable model"; break; }
    case clingo_model_type_brave_consequences:    { type_string = "Brave consequences"; break; }
    case clingo_model_type_cautious_consequences: { type_string = "Cautious consequences"; break; }
  }
  // get running number of model
  if (!clingo_model_number(model, &number)) { goto error; }
  printf("%s %" PRIu64 ":\n", type_string, number);
  if (!print_model(model, &data, "  shown", clingo_show_type_shown)) { goto error; }
  if (!print_model(model, &data, "  atoms", clingo_show_type_atoms)) { goto error; }
  if (!print_model(model, &data, "  terms", clingo_show_type_terms)) { goto error; }
  if (!print_model(model, &data, " ~atoms", clingo_show_type_complement
                                         | clingo_show_type_atoms)) { goto error; }
  goto out;
error:
  ret = false;
out:
  free_model_buffer(&data);
  return ret;
}

void print_bin(unsigned long long int x, size_t n) {
  while (n--) printf("%llu", (x >> n) % 2);
}

void undef_atom_ignore(clingo_warning_t code, const char *msg, void *data) {
  if (code == clingo_warning_atom_undefined) return;
  printf("clingo | error code %d: %s\n", code, msg);
  (void) data;
}

void raise_clingo_error(const char *msg) {
  char buffer[512];
  if (msg) {
    sprintf(buffer, "Clingo error %d: %s\n  Note: %s\n", clingo_error_code(),
        clingo_error_message(), msg);
  } else sprintf(buffer, "Clingo error %d: %s\n", clingo_error_code(), clingo_error_message());
  PyErr_SetString(PyExc_Exception, buffer);
}
