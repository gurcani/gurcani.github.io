#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:47:56 2020

@author: ogurcan
"""
import numpy as np
from mpi4py import MPI
#import matplotlib.pylab as plt

def aligned_data(shape,dtype):
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + 16, dtype = np.uint8)
    start_index = -buf.ctypes.data % 16
    return buf[start_index:start_index + nbytes].view(dtype).reshape(shape)


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

comm=MPI.COMM_WORLD
Nx,Ny=128,128
f=distarray((12,Nx,Ny))
#print("rank = ",comm.rank, "f.shape=",f.shape, "local_slice=",f.local_slice)
x,y=np.meshgrid(np.linspace(-1,1,Nx),np.linspace(-1,1,Ny),indexing='ij')
f[0,:,:]=(np.sin(4*np.pi*x+2*np.pi*y)*np.exp(-x**2/2/0.2-y**2/2/0.1))[f.local_slice[1:]]
# plt.pcolormesh(x[f.local_slice],y[f.local_slice],f,shading='auto',rasterized=True,cmap='seismic',vmin=-1,vmax=1)
# plt.axis('square')
# plt.axis([-1,1,-1,1])
# plt.show()
# plt.pause(1)