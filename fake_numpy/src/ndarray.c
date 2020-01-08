#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include "numpytypes.h"

typedef struct {
    PyObject_HEAD
    int data_size;
    int nd;
    int *dimensions;
    int *strides;
    char *data;
    char dtype;
    //Offset is required here
} PyArrayObject;

static void
PyArray_dealloc(PyArrayObject *self)
{
    free(self->dimensions);
    free(self->strides);
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
PyArray_init(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    //Add support to more args
    PyObject *tuple = NULL;
    int dtype = 0;
    char *keywords[] = {"", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", keywords,
                &tuple, &dtype)) {
        Py_XDECREF(tuple);
        return -1;
    }
    if (!PyTuple_CheckExact(tuple)) {
        PyErr_SetString(PyExc_ValueError,
                "ndarray expects tuple as the first argument");
        Py_XDECREF(tuple);
        return -1;
    }
    self->dtype = dtype;
    switch (self->dtype)
    {
        case PyArray_float64:
            self->data_size = sizeof(double);
            break;
        case PyArray_int8:
            self->data_size = 1;
            break;
        case PyArray_int16:
            self->data_size = 2;
            break;
        case PyArray_int32:
            self->data_size = 4;
            break;
        case PyArray_int64:
            self->data_size = 8;
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                    "dtype argument value not valid");
            Py_DECREF(tuple);
            return -1;
    }
    int tuple_len = (int) PyTuple_GET_SIZE(tuple);
    self->nd = tuple_len;
    self->dimensions = (int *) malloc(sizeof(int) * tuple_len);
    self->strides = (int *) malloc(sizeof(int) * tuple_len);
    if (self->dimensions == NULL || self->strides == NULL) {
        Py_DECREF(tuple);
        PyErr_NoMemory();
        return -1;
    }
    int dim_prod = 1;
    for (int pos = 0; pos < self->nd; pos++) {
        long d = PyLong_AsLong(PyTuple_GET_ITEM(tuple, (Py_ssize_t) pos));
        self->dimensions[pos] = (int) d;
        dim_prod *= self->dimensions[pos];
    }
    self->data = (char *) malloc(sizeof(char) * dim_prod * self->data_size);
    if (self->data == NULL) {
        Py_DECREF(tuple);
        PyErr_NoMemory();
        return -1;
    }
    for (int pos = self->nd - 1; pos >= 0; pos--) {
        if (pos == self->nd - 1) {
            self->strides[pos] = self->data_size;
        } else {
            self->strides[pos] = self->dimensions[pos+1] * self->strides[pos+1];
        }
    }
    return 0;
}

/* Getters and Setters */

static PyObject *
PyArray_shape(PyArrayObject *self)
{
    PyObject *list = Py_BuildValue("[]");
    if (list == NULL) {
        Py_XDECREF(list);
        return NULL;
    }
    for (int i = 0; i < self->nd; i++) {
        if (PyList_Append(list, PyLong_FromLong(self->dimensions[i])) < 0) {
            Py_XDECREF(list);
            return NULL;
        }
    }
    PyObject *tuple = PyList_AsTuple(list);
    Py_DECREF(list);
    return tuple;
}

static PyObject *
PyArray_strides(PyArrayObject *self)
{
    PyObject *list = Py_BuildValue("[]");
    if (list == NULL) {
        Py_XDECREF(list);
        return NULL;
    }
    for (int i = 0; i < self->nd; i++) {
        if (PyList_Append(list, PyLong_FromLong(self->strides[i])) < 0) {
            Py_XDECREF(list);
            return NULL;
        }
    }
    PyObject *tuple = PyList_AsTuple(list);
    Py_DECREF(list);
    return tuple;
}

static PyObject *
PyArray_ndim(PyArrayObject *self)
{
    return PyLong_FromLong((long) self->nd);
}

static PyGetSetDef PyArray_getsetters[] = {
    {"shape", (getter) PyArray_shape, NULL,
    "Returns the length of the first dimension", NULL},
    {"strides", (getter) PyArray_strides, NULL,
    "Returns the strides of the ndarray", NULL},
    {"ndim", (getter) PyArray_ndim, NULL,
    "Returns the number of dimensions", NULL},
    {NULL} /* Sentinel */
};

/* Methods */

static PyObject *
PyArray_copy(PyArrayObject *self)
{
    PyObject *args = Py_BuildValue("(Oi)", PyArray_shape(self), self->dtype);
    if (args == NULL)
        return NULL;
    PyObject *kwds = PyDict_New();
    if (kwds == NULL)
        return NULL;
    PyArrayObject *ndarray_obj = (PyArrayObject *) PyObject_Call(
            PyObject_Type((PyObject *) self), args, kwds);
    Py_DECREF(args);
    Py_DECREF(kwds);
    if (ndarray_obj == NULL)
        return NULL;
    int dim_prod = 1;
    for (int d = 0; d < ndarray_obj->nd; d++)
        dim_prod *= ndarray_obj->dimensions[d];
    for (int i = 0; i < dim_prod; i++)
        ndarray_obj->data[i] = ((PyArrayObject *) self)->data[i];
    return (PyObject *) ndarray_obj;
}

static PyMethodDef PyArray_methods[] = {
    {"copy", (PyCFunction) PyArray_copy, METH_NOARGS,
    "Returns an identical ndarray within a new memory location"},
    {NULL} /* Sentintel */
};

/* Sequence/Mapping methods */

static int
PyArray_SetItem(PyObject *self, PyObject *key, PyObject *value)
{
    
}

static PyObject *
PyArray_GetItem(PyObject *self, PyObject *value)
{
    int i = (int) PyLong_AsLong(value);
    return PyLong_FromLong((long) ((PyArrayObject *)self)->data[i]);
}

static Py_ssize_t
PyArray_len(PyObject *self)
{
    return ((PyArrayObject *) self)->dimensions[0];
}

static PyMappingMethods PyArrayMapping = {
    .mp_subscript = PyArray_GetItem,
};

static PySequenceMethods PyArraySequence = {
    .sq_length = PyArray_len,
};

static PyTypeObject PyArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ndarray.ndarray",
    .tp_basicsize = sizeof(PyArrayObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) PyArray_init,
    .tp_dealloc = (destructor) PyArray_dealloc,
    .tp_methods = PyArray_methods,
    .tp_getset = PyArray_getsetters,
    .tp_as_sequence = &PyArraySequence,
    .tp_as_mapping = &PyArrayMapping
};

/* ndarray functions */

static PyObject *
zeros(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *ndarray_obj = (PyArrayObject *) PyObject_Call(
            (PyObject *) &PyArrayType, args, kwds);
    if (ndarray_obj == NULL)
        return NULL;
    int dim_prod = 1;
    for (int d = 0; d < ndarray_obj->nd; d++)
        dim_prod *= ndarray_obj->dimensions[d];
    for (int i = 0; i < dim_prod; i++)
        ndarray_obj->data[i] = 0;
    return (PyObject *) ndarray_obj;
}

static PyObject *
ones(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *ndarray_obj = (PyArrayObject *) PyObject_Call(
            (PyObject *) &PyArrayType, args, kwds);
    if (ndarray_obj == NULL)
        return NULL;
    int dim_prod = 1;
    for (int d = 0; d < ndarray_obj->nd; d++)
        dim_prod *= ndarray_obj->dimensions[d];
    for (int i = 0; i < dim_prod; i++)
        ndarray_obj->data[i] = 1;
    return (PyObject *) ndarray_obj;
}

static PyObject *
empty(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *ndarray_obj = PyObject_Call((PyObject *) &PyArrayType, 
            args, kwds);
    if (ndarray_obj == NULL)
        return NULL;
    return ndarray_obj;
}

void
overwrite_ndarray(PyObject *obj, PyArrayObject *ndarray_obj,
        int depth, int *shape, int *pos, int current_depth)
{
    if ((int) PyObject_Length(obj) != shape[current_depth]) {
        PyErr_SetString(PyExc_ValueError, "Inconsistent sequence shape");
        return;
    } else if (current_depth == (depth - 1)) {
        //Overwrite every element in the given dimension
    } else {
        int *tmp = (int *) malloc(sizeof(int) * depth);
        if (tmp == NULL) {
            PyErr_NoMemory();
            return;
        }
        for (int i = 0; i < depth; i++)
            tmp[i] = pos[i];
        pos = tmp;
        int len = (int) PyObject_Length(obj);
        for (int i = 0; i < len; i++) {
            PyObject *i_obj = Py_BuildValue("i", i);
            if (i_obj == NULL)
                return;
            PyObject *new_obj = PyObject_GetItem(obj, i_obj);
            Py_DECREF(i_obj);
            if (new_obj == NULL)
                return;
            if (!PySequence_Check(new_obj) && !PyMapping_Check(new_obj)) {
                PyErr_SetString(PyExc_ValueError,
                        "Inconsistent sequence shape");
                Py_DECREF(new_obj);
                free(pos);
                return;
            }
            overwrite_ndarray(new_obj, ndarray_obj, depth, shape, pos,
                    current_depth + 1);
            pos[current_depth] += 1;
            Py_DECREF(new_obj);
        }
        free(pos);
    }
}

static PyObject *
array(PyObject *self, PyObject *args, PyObject *kwds)
{
    //Array interfaces not supported, only sequence-like objects
    PyObject *obj;
    int copy = 1; //0 unsupported
    char *keywords[] = {"", "copy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p", keywords,
                &obj, &copy)) {
        Py_XDECREF(obj);
        return NULL;
    }
    if (copy == 0) {
        PyErr_SetString(PyExc_NotImplementedError, "Copy has to be True");
        Py_DECREF(obj);
        return NULL;
    }
    int depth = 0;
    if (PyObject_IsInstance(obj, (PyObject *) &PyArrayType)) {
        return PyArray_copy((PyArrayObject *) obj);
    } else if (PySequence_Check(obj) || PyMapping_Check(obj)) {
        PyObject *zero = Py_BuildValue("i", 0);
        PyObject *tmp = obj;
        if (zero == NULL)
            return NULL;
        for (int i = 0;; i++) {
            PyObject *x = tmp;
            tmp = PyObject_GetItem(tmp, zero);
            if (i > 0)
                Py_DECREF(x);
            if (tmp == NULL)
                return NULL;
            if (!PySequence_Check(tmp) && !PyMapping_Check(tmp)) {
                Py_DECREF(tmp);
                break;
            }
            depth += 1;
        }
        Py_DECREF(zero);
    } else {
        PyErr_SetString(PyExc_ValueError,
                "First argument has to provied a __getitem__ method.");
        Py_DECREF(obj);
        return NULL;
    }
    if (kwds != NULL && PyDict_Contains(kwds, Py_BuildValue("s", "copy"))) {
        PyDict_DelItemString(kwds, "copy");
    }
    PyObject *ndarray_obj = PyObject_Call((PyObject *) &PyArrayType, 
            Py_BuildValue("((i))", 5), kwds);
    if (ndarray_obj == NULL)
        return NULL;
    int *shape = (int *) malloc(sizeof(int) * depth);
    PyObject *zero = Py_BuildValue("i", 0);
    PyObject *tmp = obj;
    if (zero == NULL)
        return NULL;
    for (int i = 0; i < depth; i++) {
        shape[i] = PyObject_Length(tmp);
        if (shape[i] < 0)
            return NULL;
        PyObject *x = tmp;
        tmp = PyObject_GetItem(tmp, zero);
        if (i > 0)
            Py_DECREF(x);
        if (tmp == NULL)
            return NULL;
    }
    Py_DECREF(zero);
    int *pos = (int *) malloc(sizeof(int) * depth);
    for (int i = 0; i < depth; i++)
        pos[i] = 0;
    overwrite_ndarray(obj, (PyArrayObject *)ndarray_obj, depth, shape, pos, 0);
    if (PyErr_Occurred())
        return NULL;
    free(shape);
    free(pos);
    return ndarray_obj;
}

static PyObject *
arange(PyObject *self, PyObject *args, PyObject *kwds)
{

}

static PyObject *
linspace(PyObject *self, PyObject *args, PyObject *kwds)
{

}

static PyMethodDef ndarray_funcs[] = {
    {"zeros", (PyCFunction)(void(*)(void)) zeros, METH_VARARGS | METH_KEYWORDS,
    "Return a ndarray with all zeros"},
    {"ones", (PyCFunction)(void(*)(void)) ones, METH_VARARGS | METH_KEYWORDS,
    "Return a ndarray with all ones"},
    {"empty", (PyCFunction)(void(*)(void)) empty, METH_VARARGS | METH_KEYWORDS,
    "Return an empty ndarray"},
    {"array",(PyCFunction)(void(*)(void)) array, METH_VARARGS | METH_KEYWORDS,
    "Return a ndarray based on the passed sequence"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition */

static PyModuleDef ndarraymodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ndarray",
    .m_doc = "Ndarray object module",
    .m_size = -1,
    .m_methods = ndarray_funcs
};

PyMODINIT_FUNC
PyInit_ndarray(void)
{
    PyObject *m;
    if (PyType_Ready(&PyArrayType) < 0)
        return NULL;

    m = PyModule_Create(&ndarraymodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyArrayType);
    if (PyModule_AddObject(m, "ndarray", (PyObject *) &PyArrayType) < 0) {
        Py_DECREF(&PyArrayType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
