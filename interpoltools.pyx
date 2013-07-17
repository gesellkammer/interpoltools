#cython: boundscheck=False
#cython: embedsignature=True
#cython: infer_types=True

cdef extern from "math.h":
    cdef double cos(double n)
    cdef double pow(double x, double y)
    cdef double ceil(double x)
    cdef double exp(double x)
    cdef double log(double x)
    cdef double sqrt(double x)

DEF MAX_FLOAT = 3.40282346638528860e+38
DEF E   = 2.718281828459045235360287471352662497757247093
DEF PHI = 1.61803398874989484820458683436563811772030917
DEF PI  = 3.141592653589793238462643383279502884197169399375105

cdef dict FUNCNAME_CASE = {
    'linear' :0,
    'halfcos':1,
    'expon'  :2,
    'cuadratic':3,
    'cubic':4
}

import numpy
cimport numpy
from numpy cimport ( 
    ndarray, npy_intp, PyArray_DIM, PyArray_EMPTY,PyArray_ISCONTIGUOUS, NPY_DOUBLE
)

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t

cdef inline ndarray EMPTY1D(int size):
    cdef npy_intp *dims = [size]
    return PyArray_EMPTY(1, dims, NPY_DOUBLE, 0)

# ----------------------------------
# LINEAR 
# -----------------------------------
cdef inline double _interpol_linear(double x, double x0, double y0, double x1, double y1):
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

def interpol_linear(double x, double x0, double y0, double x1, double y1):
    """
    interpolate between (x0, y0) and (x1, y1) at point x
    """
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

def ilin1(double x, double y0, double y1):
    """
    x: a number between 0-1. 0: y0, 1: y1
    """
    return _interpol_linear(x, 0, y0, 1, y1)

# ----------------------------------
# Halfcos(exp)
# -----------------------------------

def interpol_halfcos(double x, double x0, double y0, double x1, double y1):
    """
    interpolate between (x0, y0) and (x1, y1) at point x
    """
    cdef double dx
    dx = ((x - x0) / (x1 - x0)) * 3.14159265358979323846 + 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)

cdef inline double _interpol_halfcos(double x, double x0, double y0, double x1, double y1):
    cdef double dx
    dx = ((x - x0) / (x1 - x0)) * 3.14159265358979323846 + 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)

def interpol_halfcosexp(double x, double x0, double y0, double x1, double y1, double exp):
    """
    interpolate between (x0, y0) and (x1, y1) at point x

    exp defines the exponential of the curve
    """
    cdef double dx
    dx = pow((x - x0) / (x1 - x0), exp)
    dx = (dx + 1.0) * 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)

cdef inline double _interpol_halfcosexp(double x, double x0, double y0, double x1, double y1, double exp):
    cdef double dx
    dx = pow((x - x0) / (x1 - x0), exp)
    dx = (dx + 1.0) * 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)

# ----------------------------------
# Fibonacci
# -----------------------------------

cdef inline double _fib(double x):
    """
    taken from home.comcast.net/~stuartmanderson/fibonacci.pdf
    fib at x = e^(x * ln(phi)) - cos(x * pi) * e^(x * ln(phi))
               -----------------------------------------------
                                     sqrt(5)
    """
    cdef double x_mul_log_phi = x * 0.48121182505960348 # 0.48121182505960348 = log(PHI)
    return (exp(x_mul_log_phi) - cos(x * PI) * exp(-x_mul_log_phi)) / 2.23606797749978969640917366873127623544
    
def fib(double x):
    """
    taken from home.comcast.net/~stuartmanderson/fibonacci.pdf
    fib at x = e^(x * ln(phi)) - cos(x * pi) * e^(x * ln(phi))
               -----------------------------------------------
                                     sqrt(5)
    """
    return _fib(x)

def interpol_fib(double x, double x0, double y0, double x1, double y1):
    """
    fibonacci interpolation. it is assured that if x is equidistant to
    x0 and x1, then for the result y it should be true that

    y1 / y == y / y0 == ~0.618
    """
    cdef double dx = (x - x0) / (x1 - x0)
    cdef double dx2 = _fib(40 + dx * 2)
    cdef double dx3 = (dx2 - 102334155) / (165580141)
    return y0 + (y1 - y0) * dx3

def ifib1(double x, double y0, double y1):
    """
    fibonacci interpolatation within the interval 0, 1

    the same as
    >> interpol_fib(x, 0, y0, 1, y1)

    if x is negative, the interval is reversed
    >> interpol_fib(abs(x), 1, y1, 0, y0)
    """
    if x < 0:
        return _interpol_fib(x * -1, 1, y1, 0, y0)
    else:
        return _interpol_fib(x, 0, y0, 1, y1)

cdef inline double _interpol_fib(double x, double x0, double y0, double x1, double y1):
    """
    fibonacci interpolation. it is assured that if x is equidistant to
    x0 and x1, then for the result y it should be true that

    y1 / y == y / y0 == ~0.618
    """
    cdef double dx = (x - x0) / (x1 - x0)
    cdef double dx2 = _fib(40 + dx * 2)
    cdef double dx3 = (dx2 - 102334155) / (165580141)
    return y0 + (y1 - y0) * dx3

cdef inline double _fib_gen(double x, double phi):
    return (exp(x * log(phi)) - cos(x * PI) * exp(x * log(phi) * -1)) / 2.23606797749978969640917366873127623544

def fib_gen(double x, double phi=1.61803398874989484820458683436563811772030917):
    assert phi > 1.001
    return _fib_gen(x, phi)

# ----------------------------------
# Exponential
# -----------------------------------

def interpol_expon(double x, double x0, double y0, double x1, double y1, double exp):
    """
    interpolate between (x0, y0) and (x1, y1) at point x

    exp defines the exponential of the curve
    """
    cdef double dx = (x - x0) / (x1 - x0)
    return y0 + pow(dx, exp) * (y1 - y0)

cdef double _interpol_expon(double x, double x0, double y0, double x1, double y1, double exp):
    cdef double dx = (x - x0) / (x1 - x0)
    return y0 + pow(dx, exp) * (y1 - y0)

def interpol_expon_new(double expon):
    """
    create a new interpolation function with exponent = expon
    """
    return _InterpolExpon_new(expon)

cdef class _InterpolExpon:
    """
    f(x, x0, y0, x1, y1)

    return y at x interpolated between (x0, y0) and (x1, y1)
    """
    cdef double expon
    def __call__(self, double x, double x0, double y0, double x1, double y1):
        cdef double dx = (x - x0) / (x1 - x0)
        return y0 + pow(dx, self.expon) * (y1 - y0)

cdef _InterpolExpon _InterpolExpon_new(double expon):
    cdef _InterpolExpon f = _InterpolExpon()
    f.expon = expon
    return f

def interpol_cuadratic(double x, double x0, double y0, double x1, double y1):
    """
    interpolate between (x0, y0) and (x1, y1) at point x
    """
    cdef double dx = (x - x0) / (x1 - x0)
    return y0 + pow(dx, 2) * (y1 - y0)

def interpol_cubic(double x, double x0, double y0, double x1, double y1):
    """
    interpolate between (x0, y0) and (x1, y1) at point x
    """
    cdef double dx = (x - x0) / (x1 - x0)
    return y0 + pow(dx, 3) * (y1 - y0)

# ----------------------------------
# Search
# -----------------------------------

def searchsorted(numpy.ndarray[DTYPE_t, ndim=1] xs, double x):
    """
    search for x in xs assuming that xs is sorted

    xs must be a numpy array of double type (float64)
    """
    cdef int imin = 0
    cdef int imax = xs.size
    cdef int imid
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if xs[imid] < x:
            imin = imid + 1
        else:
            imax = imid
    return imin

cdef inline int _searchsorted(numpy.ndarray xs, double x):
    cdef int imin = 0
    cdef int imid
    cdef numpy.ndarray[DTYPE_t, ndim=1] _xs = xs
    #cdef int imax = _xs.size
    cdef int imax = PyArray_DIM(_xs, 0)
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if _xs[imid] < x:
            imin = imid + 1
        else:
            imax = imid
    return imin

def array_interpol_linear(numpy.ndarray[DTYPE_t, ndim=1] xs, numpy.ndarray[DTYPE_t, ndim=1] ys, double x):
    """
    xs and ys define a series of points ((xs[0], ys[0]), (xs[1], ys[1]), ... (xs[n], ys[n]))

    for a given x find the corresponding y with linear interpolation

    This is equivalent to:

    >>> from bpf4 import Linear
    >>> Linear(xs, ys)(x)

    """
    cdef int index1 = _searchsorted(xs, x)
    cdef int index0 = index1 - 1
    cdef double x0 = xs[index0]
    cdef double y0 = ys[index0]
    return y0 + (ys[index1] - y0) * ((x - x0) / (xs[index1] - x0))

def interpol_between(double a, double b, int num=50, str func='linear', endpoint=True):
    """
    generate an array shape=(num,) interpolating between a and b with function func

    func can be:
        'linear'
        'halfcos'
        'cuadratic'
        'cubic'
        'expon(2.0)'  == cuadratic
        'halfcos(0.5)'

    if func is left to its default 'linear', this is a drop-in replacement for numpy.linspace

    with func you can specify the shape of the interpolation
    """
    cdef numpy.ndarray[DTYPE_t, ndim=1] xs = numpy.empty((num,), dtype=DTYPE)
    cdef int i
    cdef double dx
    cdef double exp
    cdef double x
    cdef int case
    if "(" in func:
        func, exps = func.split("(")
        exp = float(exps[:-1])
    if not endpoint:
        dx = 1.0 / <double>num
    else:
        dx = 1.0 / <double>(num - 1)
    case = FUNCNAME_CASE.get(func)
    if case == 0:   # linear
        for i in range(num):
            x = dx * i
            xs[i] = _interpol_linear(x, 0, a, 1, b)
    elif case == 1: # halfcos
        for i in range(num):
            x = dx * i
            xs[i] = _interpol_halfcos(x, 0, a, 1, b)
    elif case == 2: # expon
        for i in range(num):
            x = dx * i
            xs[i] = _interpol_expon(x, 0, a, 1, b, exp)
    elif case == 3: # cuadratic
        for i in range(num):
            x = dx * i
            xs[i] = _interpol_expon(x, 0, a, 1, b, 2)
    elif case == 4: # cubic
        for i in range(num):
            x = dx * i
            xs[i] = _interpol_expon(x, 0, a, 1, b, 3)
    return xs