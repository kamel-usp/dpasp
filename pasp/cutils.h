#ifndef Py_CUTILSMODULE_H
#define Py_CUTILSMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <clingo.h>

typedef struct string {
  char *s;
  size_t n;
} string_t;

#define PyCutils_string_from_symbol_NUM 0
#define PyCutils_string_from_symbol_RETURN bool
#define PyCutils_string_from_symbol_PROTO (clingo_symbol_t sym, string_t *buf)

#define PyCutils_string_free_NUM 1
#define PyCutils_string_free_RETURN void
#define PyCutils_string_free_PROTO (string_t *s)

#define PyCutils_print_solution_NUM 2
#define PyCutils_print_solution_RETURN bool
#define PyCutils_print_solution_PROTO (const clingo_model_t *model)

#define PyCutils_print_bin_NUM 3
#define PyCutils_print_bin_RETURN void
#define PyCutils_print_bin_PROTO (unsigned long long int x, size_t n)

#define PyCutils_undef_atom_ignore_NUM 4
#define PyCutils_undef_atom_ignore_RETURN void
#define PyCutils_undef_atom_ignore_PROTO (clingo_warning_t code, const char *msg, void *data)

#define PyCutils_char_from_symbol_NUM 5
#define PyCutils_char_from_symbol_RETURN bool
#define PyCutils_char_from_symbol_PROTO (clingo_symbol_t sym, char *s, size_t n)

#define PyCutils_API_pointers 6

#ifdef CUTILS_MODULE

static PyCutils_string_from_symbol_RETURN string_from_symbol PyCutils_string_from_symbol_PROTO;
static PyCutils_string_free_RETURN string_free PyCutils_string_free_PROTO;
static PyCutils_print_solution_RETURN print_solution PyCutils_print_solution_PROTO;
static PyCutils_print_bin_RETURN print_bin PyCutils_print_bin_PROTO;
static PyCutils_undef_atom_ignore_RETURN undef_atom_ignore PyCutils_undef_atom_ignore_PROTO;
static PyCutils_char_from_symbol_RETURN char_from_symbol PyCutils_char_from_symbol_PROTO;

#else

static void** PyCutils_API;

#define string_from_symbol \
  (*(PyCutils_string_from_symbol_RETURN (*)PyCutils_string_from_symbol_PROTO) PyCutils_API[PyCutils_string_from_symbol_NUM])
#define string_free \
  (*(PyCutils_string_free_RETURN (*)PyCutils_string_free_PROTO) PyCutils_API[PyCutils_string_free_NUM])
#define print_solution \
  (*(PyCutils_print_solution_RETURN (*)PyCutils_print_solution_PROTO) PyCutils_API[PyCutils_print_solution_NUM])
#define print_bin \
  (*(PyCutils_print_bin_RETURN (*)PyCutils_print_bin_PROTO) PyCutils_API[PyCutils_print_bin_NUM])
#define undef_atom_ignore \
  (*(PyCutils_undef_atom_ignore_RETURN (*)PyCutils_undef_atom_ignore_PROTO) PyCutils_API[PyCutils_undef_atom_ignore_NUM])
#define char_from_symbol \
  (*(PyCutils_char_from_symbol_RETURN (*)PyCutils_char_from_symbol_PROTO) PyCutils_API[PyCutils_char_from_symbol_NUM])

static int import_cutils(void) {
  PyCutils_API = (void**) PyCapsule_Import("cutils._C_API", 0);
  return (PyCutils_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
