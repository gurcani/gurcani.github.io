#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 02:08:16 2020

@author: ogurcan
"""
from numba import njit
import numpy as np
from mpi4py import MPI
import sys
sys.path.insert(1, '/home/ogurcan/working/hwak')
from hwdistarray import hwdistarray

# @njit(fastmath=True)
# def compravel(v,u):
#     k=0
#     Nx=u.shape[1]
#     Ny=u.shape[2]
#     for l in range(u.shape[0]):
#         for i in range(int(Nx/2)):
#             for j in range(Ny-1):
#                 v[k]=u[l,i,j]
#                 k+=1
#         for i in range(int(Nx/2+1),Nx):
#             for j in range(1,Ny-1):
#                 v[k]=u[l,i,j]
#                 k+=1

def compravel(v,u):
    last=(u.local_slice[2].stop==u.global_shape[2])
    first=(u.local_slice[2].start==0)
    compravel_(v,u,u.shape[2],first,last)

@njit(fastmath=True)
def compravel_(v,u,Ny,first,last):
    k=0
    Nx=u.shape[1]
    for l in range(u.shape[0]):
        for i in range(int(Nx/2)):
            for j in range(Ny-int(last)):
                v[k]=u[l,i,j]
                k+=1
        for i in range(int(Nx/2+1),Nx):
            for j in range(int(first),Ny-int(last)):
                v[k]=u[l,i,j]
                k+=1

def expunravel(v,u):
    last=(u.local_slice[2].stop==u.global_shape[2])
    first=(u.local_slice[2].start==0)
    expunravel_(u,v,u.shape[2],first,last)

# @njit(fastmath=True)
# def expunravel2_(u,v,Ny,ny0s):
#     k=0
#     Nx=u.shape[1]
#     for l in range(u.shape[0]):
#         for i in range(int(Nx/2)):
#             for j in range(Ny):
#                 u[l,i,j]=v[k]
#                 k+=1
#         for i in range(int(Nx/2+1),Nx):
#             if(ny0s): u[l,i,0]=u[l,Nx-i,0].real-1j*u[l,Nx-i,0].imag
#             for j in range(ny0s,Ny):
#                 u[l,i,j]=v[k]
#                 k+=1

@njit(fastmath=True)
def expunravel_(u,v,Ny,first,last):
    k=0
    Nx=u.shape[1]
    for l in range(u.shape[0]):
        for i in range(int(Nx/2)):
            for j in range(Ny-int(last)):
                u[l,i,j]=v[k]
                k+=1
            if last: u[l,i,-1]=0
        for j in range(Ny): u[l,int(Nx/2),j]=0
        for i in range(int(Nx/2+1),Nx):
            if first: u[l,i,0]=u[l,Nx-i,0].real-1j*u[l,Nx-i,0].imag
            for j in range(int(first),Ny-int(last)):
                u[l,i,j]=v[k]
                k+=1
            if last : u[l,i,-1]=0

Nx,Ny=8,8
comm=MPI.COMM_WORLD
uk=hwdistarray((2,Nx,int(Ny/2+1)),comm=comm,dtype=complex)
inds=np.r_[int(Nx/2):Nx,0:int(Nx/2)]
uk[0,:,:]=np.array([[inds[l-1]+1j*m for m in np.r_[uk.local_slice[2]]] for l in np.r_[uk.local_slice[1]]])
uk[1,:,:]=np.array([[2*inds[l-1]+1j*(m+1) for m in np.r_[uk.local_slice[2]]] for l in np.r_[uk.local_slice[1]]])

#if(uk.local_slice[2].stop==uk.global_shape[2]):
#    uk[:,:,-1]=0
#uk[:,int(Nx/2),:]=0

last=(uk.local_slice[2].stop==uk.global_shape[2])
first=(uk.local_slice[2].start==0)
Nv=uk.shape[0]*((uk.shape[1]-1)*(uk.shape[2]-int(last))-int(first)*int(Nx/2-1))
v=np.zeros(Nv,dtype=complex)
#v=hwdistarray((((Nx-1)*(int(Ny/2))-int(Nx/2-1))*uk.shape[0],),dtype=complex)
#print(uk.shape,v.shape)
compravel(v,uk)
print(uk[0,],v[:int(v.shape[0]/2)])
up=hwdistarray((2,Nx,int(Ny/2+1)),comm=comm,dtype=complex)
#up[:]=0
expunravel(v,up)
print(up[0,])
