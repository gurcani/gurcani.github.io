import numpy as np

n=20

f=np.array([l+1j*(l+1) for l in range(n)])
g=np.array([l+1j*(2*l+1) for l in range(n)])
h=np.array([l+1j*(3*l+1) for l in range(n)])
r=np.array([l+1j*(4*l+1) for l in range(n)])
res=np.zeros(r.shape)

F=(f,g,h,r)

from ctypes import cdll,c_uint,c_double,POINTER
from numpy.ctypeslib import ndpointer

libcpy = cdll.LoadLibrary('./libcpytest.so')
libcpy.testabssum.argtypes=[POINTER(POINTER(c_double)),ndpointer(dtype=float),c_uint,c_uint]

Fp=(POINTER(c_double)*len(F))(*[l.ctypes.data_as(POINTER(c_double)) for l in F])

libcpy.testabssum(Fp,res,n,len(F))
print("res=",res)

# res2=np.sum(np.abs(np.array(F)),axis=0)
# print(res2)