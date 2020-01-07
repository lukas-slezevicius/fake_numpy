#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/* The function code

static PyObject *
test_func(PyObject *self, PyObject *args)
{
    const int n;
    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;
    return PyLong_FromLong(n * 2);
}

static PyMethodDef NdarrayMethods[] = {
    {"test_func", test_func, METH_VARARGS, "Multiply by 2"}
};

static struct PyModuleDef ndarraymodule = {
    PyModuleDef_HEAD_INIT,
    "ndarray",
    NULL,
    -1,
    NdarrayMethods
};

PyMODINIT_FUNC
PyInit_ndarray(void)
{
    return PyModule_Create(&ndarraymodule);
}

*/

typedef struct {
    PyObject_HEAD
    int data_size;
    int nd;
    int *dimensions;
    char *data;
} PyArrayObject;

static PyTypeObject PyArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ndarray.ndarray",
    .tp_doc = "Ndarray object",
    .tp_basicsize = sizeof(PyArrayObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};

/* use sentintel values! */
