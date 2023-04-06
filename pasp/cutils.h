#ifndef _PASP_CUTILS
#define _PASP_CUTILS

#include <clingo.h>

extern const clingo_part_t GROUND_DEFAULT_PARTS[];
extern const char* const CONTROL_DEFAULT_ARGS[];
extern const size_t CONTROL_DEFAULT_NARGS;

typedef struct string {
  char *s;
  size_t n;
} string_t;

bool string_from_symbol(clingo_symbol_t sym, string_t *buf);
void string_free(string_t *s);
bool print_solution(const clingo_model_t *model);
void print_bin(unsigned long long int x, size_t n);
void undef_atom_ignore(clingo_warning_t code, const char *msg, void *data);

void raise_clingo_error(const char *msg);

#endif
