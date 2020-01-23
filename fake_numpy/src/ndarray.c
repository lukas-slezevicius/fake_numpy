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

int
PyArray_AssignDataSize(PyArrayObject *self)
{
    switch (self->dtype)
    {
        case PyArray_float64:
            self->data_size = sizeof(double);
            return 0;
        case PyArray_int8:
            self->data_size = 1;
            return 0;
        case PyArray_int16:
            self->data_size = 2;
            return 0;
        case PyArray_int32:
            self->data_size = 4;
            return 0;
        case PyArray_int64:
            self->data_size = 8;
            return 0;
        default:
            PyErr_SetString(PyExc_ValueError,
                    "dtype argument value not valid");
            return -1;
    }
}

void
PyArray_AssignStrides(PyArrayObject *self)
{
    for (int pos = self->nd - 1; pos >= 0; pos--) {
        if (pos == self->nd - 1) {
            self->strides[pos] = self->data_size;
        } else {
            self->strides[pos] = self->dimensions[pos+1] * self->strides[pos+1];
        }
    }
}

int
PyArray_UpdateDataType(PyArrayObject *self, int dtype)
{
    int old_dtype = dtype;
    self->dtype = dtype;
    if (PyArray_AssignDataSize(self) < 0) {
        self->dtype = old_dtype;
        return -1;
    }
    PyArray_AssignStrides(self);
    return 0;
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
        Py_DECREF(tuple);
        return -1;
    }
    self->dtype = dtype;
    if (PyArray_AssignDataSize(self) < 0) {
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
    PyArray_AssignStrides(self);
    return 0;
}

/* Getters and Setters */

static PyObject *
PyArray_shape(PyArrayObject *self)
{
    PyObject *list = Py_BuildValue("[]");
    if (list == NULL) {
        return NULL;
    }
    for (int i = 0; i < self->nd; i++) {
        if (PyList_Append(list, PyLong_FromLong(self->dimensions[i])) < 0) {
            Py_DECREF(list);
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
        return NULL;
    }
    for (int i = 0; i < self->nd; i++) {
        if (PyList_Append(list, PyLong_FromLong(self->strides[i])) < 0) {
            Py_DECREF(list);
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
        ndarray_obj->data[i] = self->data[i];
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
    int i = (int) PyLong_AsLong(value) * ((PyArrayObject *)self)->data_size;
    switch (((PyArrayObject *)self)->dtype) {
        case 0:
            {
            double tmp_val;
            memcpy(&tmp_val, &((PyArrayObject *)self)->data[i], sizeof(double));
            return PyFloat_FromDouble(tmp_val);
            }
        case 1:
            {
            int8_t tmp_val;
            memcpy(&tmp_val, &((PyArrayObject *)self)->data[i], sizeof(int8_t));
            return PyLong_FromLong(tmp_val);
            }
        case 2:
            {
            int16_t tmp_val;
            memcpy(&tmp_val, &((PyArrayObject *)self)->data[i], sizeof(int16_t));
            return PyLong_FromLong(tmp_val);
            }
        case 3:
            {
            int32_t tmp_val;
            memcpy(&tmp_val, &((PyArrayObject *)self)->data[i], sizeof(int32_t));
            return PyLong_FromLong(tmp_val);
            }
        case 4:
            {
            int64_t tmp_val;
            memcpy(&tmp_val, &((PyArrayObject *)self)->data[i], sizeof(int64_t));
            return PyLong_FromLong(tmp_val);
            }
        default:
            PyErr_SetString(PyExc_SystemError, "Data type incorrect");
            return NULL;
    }
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
        /* The object at a given depth does not match the shape */
        PyErr_SetString(PyExc_ValueError, "Inconsistent sequence shape");
        return;
    } else if (current_depth == (depth - 1)) {
        /* Write the obj values at the last depth to the ndarray */
        int data_idx = 0;
        for (int i = 0; i < depth - 1; i++) //Index based on strides
            data_idx += pos[i] * ndarray_obj->strides[i];
        for (int i = 0; i < shape[depth - 1]; i++) {
            PyObject *i_obj = Py_BuildValue("i", i);
            if (i_obj == NULL) {
                return;
            }
            PyObject *new_obj = PyObject_GetItem(obj, i_obj);
            Py_DECREF(i_obj);
            if (new_obj == NULL) {
                return;
            }
            switch (ndarray_obj->dtype) {
                case 0:
                    {
                    double tmp_val = PyFloat_AsDouble(new_obj);
                    memcpy(ndarray_obj->data + data_idx,
                            &tmp_val, sizeof(double));
                    break;
                    }
                case 1:
                    {
                    int8_t tmp_val = PyLong_AsLong(new_obj);
                    memcpy(ndarray_obj->data + data_idx,
                            &tmp_val, sizeof(int8_t));
                    break;
                    }
                case 2:
                    {
                    int16_t tmp_val = PyLong_AsLong(new_obj);
                    memcpy(ndarray_obj->data + data_idx,
                            &tmp_val, sizeof(int16_t));
                    break;
                    }
                case 3:
                    {
                    int32_t tmp_val = PyLong_AsLong(new_obj);
                    memcpy(ndarray_obj->data + data_idx,
                            &tmp_val, sizeof(int32_t));
                    break;
                    }
                case 4:
                    {
                    int64_t tmp_val = PyLong_AsLong(new_obj);
                    memcpy(ndarray_obj->data + data_idx,
                            &tmp_val, sizeof(int64_t));
                    break;
                    }
                default:
                    PyErr_SetString(PyExc_SystemError, "Data type incorrect");
                    Py_DECREF(new_obj);
                    return;
            }
            Py_DECREF(new_obj);
            data_idx += ndarray_obj->data_size;
        }
    } else {
        /* Traverse to deeper depths if it's a sequence */
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
            if (i_obj == NULL) {
                free(pos);
                return;
            }
            PyObject *new_obj = PyObject_GetItem(obj, i_obj);
            Py_DECREF(i_obj);
            if (new_obj == NULL) {
                free(pos);
                return;
            }
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
    PyObject *obj;
    int dtype = -1;
    int copy = 1;
    char *keywords[] = {"", "dtype", "copy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ip", keywords,
                &obj, &dtype, &copy)) {
        Py_XDECREF(obj);
        return NULL;
    }

    /* Check if the provided obj is an ndarray or not sequence-like */

    if (PyObject_IsInstance(obj, (PyObject *) &PyArrayType)) {
        if (copy == 1) {
            PyArrayObject *tmp = (PyArrayObject *) PyArray_copy(
                    (PyArrayObject *) obj);
            if (tmp == NULL)
                return NULL;
            if (dtype >= 0) {
                if (PyArray_UpdateDataType(tmp, dtype) < 0)
                    return NULL;
            }
            return (PyObject *) tmp;
        } else {
            return obj;
        }
    } else if (!PySequence_Check(obj) && !PyMapping_Check(obj)) {
        PyErr_SetString(PyExc_ValueError,
                "First argument has to provide a __getitem__ method.");
        Py_DECREF(obj);
        return NULL;
    }

    /* Get the depth */

    int depth = 0;
    PyObject *zero = Py_BuildValue("i", 0);
    PyObject *tmp = obj;
    if (zero == NULL)
        return NULL;
    for (int i = 0;; i++) {
        PyObject *x = tmp;
        tmp = PyObject_GetItem(tmp, zero);
        if (i > 0)
            Py_DECREF(x);
        if (tmp == NULL) {
            Py_DECREF(zero);
            return NULL;
        }
        depth += 1;
        if (!PySequence_Check(tmp) && !PyMapping_Check(tmp)) {
            Py_DECREF(zero);
            Py_DECREF(tmp);
            break;
        }
    }
    if (kwds != NULL && PyDict_Contains(kwds, Py_BuildValue("s", "copy"))) {
        if (PyDict_DelItemString(kwds, "copy") < 0) {
            Py_DECREF(zero);
            return NULL;
        }
    }

    /* Get the shape */

    int *shape = (int *) malloc(sizeof(int) * depth);
    PyObject *shape_list = Py_BuildValue("[]");
    if (shape_list == NULL) {
        Py_DECREF(zero);
        return NULL;
    }
    tmp = obj;
    for (int i = 0; i < depth; i++) {
        shape[i] = (int) PyObject_Length(tmp);
        if (shape[i] < 0) {
            Py_DECREF(zero);
            Py_DECREF(shape_list);
            return NULL;
        }
        if (PyList_Append(shape_list, PyLong_FromLong(shape[i])) < 0) {
            Py_DECREF(shape_list);
            Py_DECREF(zero);
            return NULL;
        }
        PyObject *x = tmp;
        tmp = PyObject_GetItem(tmp, zero);
        if (i > 0)
            Py_DECREF(x);
        if (tmp == NULL) {
            Py_DECREF(zero);
            Py_DECREF(shape_list);
            return NULL;
        }
    }
    Py_DECREF(zero);
    
    /* Crate the new ndarray object */

    PyObject *shape_tuple = PyList_AsTuple(shape_list);
    Py_DECREF(shape_list);
    if (shape_tuple == NULL)
        return NULL;
    PyObject *ndarray_obj = PyObject_Call((PyObject *) &PyArrayType, 
            Py_BuildValue("(O)", shape_tuple), kwds);
    Py_DECREF(shape_tuple);
    if (ndarray_obj == NULL)
        return NULL;

    /* Write the values of obj to the new ndarray */

    int *pos = (int *) malloc(sizeof(int) * depth);
    for (int i = 0; i < depth; i++)
        pos[i] = 0;
    overwrite_ndarray(obj, (PyArrayObject *)ndarray_obj, depth, shape, pos, 0);
    free(shape);
    free(pos);
    if (PyErr_Occurred()) {
        Py_DECREF(ndarray_obj);
        return NULL;
    }
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
