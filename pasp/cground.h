#ifndef Py_CGROUNDMODULE_H
#define Py_CGROUNDMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#define PyCground_ground_NUM 0
#define PyCground_ground_RETURN bool
#define PyCground_ground_PROTO (program_t *P)

#define PyCground_needs_ground_NUM 1
#define PyCground_needs_ground_RETURN bool
#define PyCground_needs_ground_PROTO (program_t *P)

#define PyCground_API_pointers 2

#ifdef CGROUND_MODULE

static PyCground_ground_RETURN ground PyCground_ground_PROTO;
static PyCground_needs_ground_RETURN needs_ground PyCground_needs_ground_PROTO;

#else

static void** PyCground_API;

#define ground \
  (*(PyCground_ground_RETURN (*)PyCground_ground_PROTO) PyCground_API[PyCground_ground_NUM])
#define needs_ground \
  (*(PyCground_needs_ground_RETURN (*)PyCground_needs_ground_PROTO) PyCground_API[PyCground_needs_ground_NUM])

static int import_cground(void) {
  PyCground_API = (void**) PyCapsule_Import("cground._C_API", 0);
  return (PyCground_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
