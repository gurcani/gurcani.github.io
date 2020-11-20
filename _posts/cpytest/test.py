import numpy as np
from ctypes import cdll,c_int,c_void_p,c_double,CFUNCTYPE,POINTER
from numpy.ctypeslib import ndpointer

def fntest(y,t):
    print('test function called')
    print('t=',t)
    return np.abs(y)**2*np.exp(t)

def fntest_p(y_p,t,n,res_p):
    y=np.ctypeslib.as_array(y_p,shape=(2*n,)).view(dtype=complex).reshape((n,))
    res=np.ctypeslib.as_array(res_p,shape=(n,))
    res[:]=fntest(y,t)

cmpfunc=CFUNCTYPE(None,POINTER(c_double), c_double, c_int, POINTER(c_double))
ftest = cmpfunc(fntest_p)
libcpy = cdll.LoadLibrary('./libcpytest.so')
libcpy.init_pars.argtypes=[c_int,cmpfunc]
libcpy.init_pars.restype=c_void_p

libcpy.fcpytest.argtypes=[c_void_p,ndpointer(dtype=complex), c_double, ndpointer(dtype=float)]

N=20
cptr=libcpy.init_pars(N,ftest)
x=np.linspace(-np.pi,np.pi,N)
y=np.exp(1j*x)
res=np.zeros(N)
t=0.5
libcpy.fcpytest(cptr,y,t,res)
print(res)