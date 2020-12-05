---
layout : post
title : Distributed Arrays in Python using mpi4py
date : 2020-12-04 22:59:00 CET
math: true
comments: true
categories : python fftw++ coding turbulence
tags : python fftw++ coding turbulence
excerpt_separator : <!--more-->
image : distarray.svg
---

[mpi4py](https://mpi4py.readthedocs.io/en/stable/) is the standard framework for using mpi under python that provides all kinds of functionality. Here loosely following the [DistArray](https://mpi4py-fft.readthedocs.io/en/latest/_modules/mpi4py_fft/distarray.html) class which is part of the [mpi4py-fft](https://mpi4py-fft.readthedocs.io), we develop a minimalistic distributed array class. Basic idea is that we can [subclass the ndarray class](https://numpy.org/doc/stable/user/basics.subclassing.html) into a distributed array class that contains:

- the local data.
- information about what part of the global array this data represents.

<!--more-->

## The DistArray Class Split Along a Single Axis:

Here we want to split an arbitrary shaped array along a selected direction. As discussed in an [earlier post]({% post_url 2020-11-26-fftw++3 %}) there are multiple ways of doing this. Here we will follow the convention which tries to 
distribute larger chunks first (i.e. the one used in fftw++). Here is the basic function that achieves this:

[**distarray.py**](https://github.com/gurcani/gurcani.github.io/tree/master/assets/examples/distarray.py))
```py
import numpy as np
from mpi4py import MPI

def localdimension(shape, rank, size,axis=-1, Nsp=0):
    sp=list(shape)
    if(Nsp==0):
        Nsp=sp[axis]
    n=int((Nsp+size-1)/size)
    start=n*rank
    extra=Nsp-start
    if(extra < 0):
        extra=0
    if(n > extra):
        n=extra
    sp[axis]=n
    off=np.zeros(len(sp),dtype=int)
    off[axis]=start
    sp=tuple(sp)
    off=tuple(off)
    return sp,off
```

This function computes the local shape **sp** and the offset **off** of the local array, of a distributed array of global shape: **shape** given the number of nodes (i.e. **size**) for a given **rank**. In order to use a properly aligned buffer( for sse2 for example) we also need an aligned allocation function, which can be written in python as:

```py
def aligned_data(shape,dtype):
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + 16, dtype = np.uint8)
    start_index = -buf.ctypes.data % 16
    return buf[start_index:start_index + nbytes].view(dtype).reshape(shape)
```

Finally we can define a class based on ndarray as:

```py
class distarray(np.ndarray):
    def __new__(self,shape,dtype=float,buffer=None,
                offset=0,strides=None,order=None,
                axis=-1,Nsp=0,comm=MPI.COMM_WORLD):
        dims=len(shape)
        locshape,loc0=localdimension(shape, comm.rank, comm.size, axis, Nsp)
        if(buffer==None):
            buffer=aligned_data(locshape,dtype)
        else:
            if(dtype!=buffer.dtype):
                print("dtype!=buffer.dtype, ignoring dtype argument")
        dtype=buffer.dtype
        obj=super(distarray, self).__new__(self,locshape,dtype,buffer,offset,strides,order)
        obj.loc0=loc0
        obj.global_shape=shape
        obj.local_slice = tuple([slice(loc0[l],loc0[l]+locshape[l],None) for l in range(dims)])
        return obj
```

The point of defining such a class is that

- we can use it directly in numpy functions as an ndarray.
- it keeps track of which part of the global array this local array represents through **loc0**, **global_shape**, **locshape**
- it also provides a direct **local_slice** that we can use on a similarly shaped global array to access only that part of the array which corresponds to the local array.

How a 2D array using a function made up of sines and exponentials is split between 4 nodes can be seen in the figure below.

[![distarray](/assets/images/distarray.svg)](/assets/images/distarray.svg)

We can do this by:

```py
comm=MPI.COMM_WORLD
Nx,Ny=128,128
f=distarray((12,Nx,Ny))
print("rank = ",comm.rank, "f.shape=",f.shape, "local_slice=",f.local_slice)
x,y=np.meshgrid(np.linspace(-1,1,Nx),np.linspace(-1,1,Ny),indexing='ij')
f[0,:,:]=(np.sin(4*np.pi*x+2*np.pi*y)*np.exp(-x**2/2/0.2-y**2/2/0.1))[f.local_slice[1:]]
```

when we run this with **mpirun -np 4 distarray.py**, it gives:
```
rank =  3 f.shape= (12, 128, 32)
rank =  2 f.shape= (12, 128, 32)
rank =  0 f.shape= (12, 128, 32)
rank =  1 f.shape= (12, 128, 32)
```

since by default it splits the array in the last dimension, the global array of shape **(12,128,128)** is divided into the 4 nodes as 4 local arrays each of which is of the shape (12,128,32).
