#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

void zc_fortran(
    const double *f,
    int nbands,
    int nlat,
    const double *lat,
    double lat_uncertainty,
    double *zc
);

static PyObject *zero_crossing(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *field_obj = NULL;
    PyObject *lat_obj = NULL;
    double lat_uncertainty = 0.0;
    PyArrayObject *field_arr = NULL;
    PyArrayObject *lat_arr = NULL;
    PyArrayObject *out_arr = NULL;
    npy_intp out_dims[1];
    static char *kwlist[] = {"field", "lat", "lat_uncertainty", NULL};

    (void) self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOd",
            kwlist,
            &field_obj,
            &lat_obj,
            &lat_uncertainty
        )) {
        return NULL;
    }

    field_arr = (PyArrayObject *) PyArray_FROM_OTF(
        field_obj, NPY_FLOAT64, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS
    );
    if (field_arr == NULL) {
        return NULL;
    }

    lat_arr = (PyArrayObject *) PyArray_FROM_OTF(
        lat_obj, NPY_FLOAT64, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS
    );
    if (lat_arr == NULL) {
        Py_DECREF(field_arr);
        return NULL;
    }

    if (PyArray_NDIM(field_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "field must be a 2D float64 array");
        goto fail;
    }
    if (PyArray_NDIM(lat_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "lat must be a 1D float64 array");
        goto fail;
    }
    if (PyArray_DIM(field_arr, 1) != PyArray_DIM(lat_arr, 0)) {
        PyErr_SetString(
            PyExc_ValueError,
            "field second dimension must match the size of lat"
        );
        goto fail;
    }

    out_dims[0] = PyArray_DIM(field_arr, 0);
    out_arr = (PyArrayObject *) PyArray_SimpleNew(1, out_dims, NPY_FLOAT64);
    if (out_arr == NULL) {
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    zc_fortran(
        (const double *) PyArray_DATA(field_arr),
        (int) PyArray_DIM(field_arr, 0),
        (int) PyArray_DIM(field_arr, 1),
        (const double *) PyArray_DATA(lat_arr),
        lat_uncertainty,
        (double *) PyArray_DATA(out_arr)
    );
    Py_END_ALLOW_THREADS

    Py_DECREF(field_arr);
    Py_DECREF(lat_arr);
    return (PyObject *) out_arr;

fail:
    Py_XDECREF(field_arr);
    Py_XDECREF(lat_arr);
    Py_XDECREF(out_arr);
    return NULL;
}

static PyMethodDef module_methods[] = {
    {
        "zero_crossing",
        (PyCFunction) (void (*)(void)) zero_crossing,
        METH_VARARGS | METH_KEYWORDS,
        "Compute the first zero crossing along the last retained latitude axis."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_zero_crossing_backend",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__zero_crossing_backend(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
