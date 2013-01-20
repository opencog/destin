
/* let the user use python list for unsigned int [] arguments */
%typemap(in)  unsigned int [] {
  // Check if is a list
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (uint *) malloc( size * sizeof(uint));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyInt_Check(o)){
            $1[i] = PyInt_AsLong(PyList_GetItem($input,i));
      } else {
            PyErr_SetString(PyExc_TypeError,"list must contain integers");
            free($1);
            return NULL;
      }
    }
    $1[i] = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

// This cleans up the uint * array we malloced before the function call
%typemap(freearg) unsigned int[] {
  free($1);
}

// let the user use python list for float [] arguments
%typemap(in)  float [] {
  // Check if is a list 
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (float *) malloc( size * sizeof(float));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyFloat_Check(o)||PyInt_Check(o)){
            $1[i] = (float)PyFloat_AsDouble(PyList_GetItem($input,i));
      } else {
            PyErr_SetString(PyExc_TypeError,"list must contain numbers");
            free($1);
            return NULL;
      }
    }
    $1[i] = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

// This cleans up the float * array we malloced before the function call
%typemap(freearg) float [] {
  free($1);
}

