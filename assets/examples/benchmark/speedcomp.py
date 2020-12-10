#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:38:15 2020

@author: ogurcan
"""

from numba import njit
import numpy as np
from time import time
from ctypes import cdll,c_uint
from numpy.ctypeslib import ndpointer
from mpi4py import MPI
from os import path
import sys

sys.path.insert(1, path.realpath(path.dirname(path.abspath(__file__))+'/../'))
from distarray import distarray

comm=MPI.COMM_WORLD

mx,my=1024,1024
rand = lambda x : np.random.rand(*x.shape)+1j*np.random.rand(*x.shape)
u=distarray((2,2*mx-1,my),dtype=complex)
a=distarray((2,)+u.global_shape,dtype=complex)
b=distarray((2,2*mx-1,my),dtype=complex)
u[:]=rand(u)
a[:]=rand(a)
b[:]=rand(b)

def mult1(v,a,b,res):
    res[:]=np.einsum('jikl,ikl->jkl',a,v)+b

def mult2(v,a,b,res):
    res[:]=np.sum(np.multiply(a,u),1)+b

@njit(fastmath=True)
def mult3(v,a,b,res):
    for j in range(v.shape[0]):
        for lx in range(v.shape[1]):
            for ly in range(v.shape[2]):
                res[j,lx,ly]=a[j,0,lx,ly]*v[0,lx,ly]+a[j,1,lx,ly]*v[1,lx,ly]+b[j,lx,ly]


libmult = cdll.LoadLibrary('./libmult.so')

libmult.multc.argtypes=[ndpointer(dtype=complex),ndpointer(dtype=complex),
                       ndpointer(dtype=complex),ndpointer(dtype=complex),c_uint]

multc=libmult.multc

N=10

res1=distarray((2,2*mx-1,my),dtype=complex)
mult3(u,a,b,res1)

if(comm.rank==0):
    dtl=[]
    dtm=[]
    print("mult1:")
for l in range(N):
    if(comm.rank==0):
        ct=time()
    mult1(u,a,b,res1)
    if(comm.rank==0):
        dt=time()-ct
        dtl.append(dt)
if(comm.rank==0):
    print ('dt=',np.mean(dtl),"+-",np.std(dtl), 'in', N, 'runs')
    dtm.append((np.mean(dtl),np.std(dtl)))
    print("mult2:")
    dtl=[]

res2=distarray((2,2*mx-1,my),dtype=complex)
for l in range(N):
    if(comm.rank==0):
        ct=time()
    mult2(u,a,b,res2)
    if(comm.rank==0):
        dt=time()-ct
        dtl.append(dt)

res3=distarray((2,2*mx-1,my),dtype=complex)
if(comm.rank==0):
    print ('dt=',np.mean(dtl),"+-",np.std(dtl), 'in', N, 'runs')
    dtm.append((np.mean(dtl),np.std(dtl)))
    print("mult3:")
    dtl=[]
for l in range(N):
    if(comm.rank==0):
        ct=time()
    mult3(u,a,b,res3)
    if(comm.rank==0):
        dt=time()-ct
        dtl.append(dt)

res4=distarray((2,2*mx-1,my),dtype=complex)
sz=np.prod(u.shape[1:])
if(comm.rank==0):
    print ('dt=',np.mean(dtl),"+-",np.std(dtl), 'in', N, 'runs')
    dtm.append((np.mean(dtl),np.std(dtl)))
    print("multc:")
    dtl=[]
for l in range(N):
    if(comm.rank==0):
        ct=time()
    multc(u,a,b,res4,sz)
    if(comm.rank==0):
        dt=time()-ct
        dtl.append(dt)
if(comm.rank==0):
    print ('dt=',np.mean(dtl),"+-",np.std(dtl), 'in', N, 'runs')
    dtm.append((np.mean(dtl),np.std(dtl)))
    dtm=np.array(dtm)
#    np.save('dtm'+str(comm.size)+'.npy',dtm)
    n=comm.size
    flname='dtm.npz'
    if(path.isfile(flname)):
        fl=np.load(flname)
        nold=fl['n']
        dtmold=fl['dtm']
        if(len(dtmold.shape)==2):
            dtm=np.stack((dtmold,dtm))
            n=np.stack((nold,n))
        else:
            dtm=np.vstack((dtmold,dtm.reshape((1,)+dtm.shape)))
            n=np.hstack((nold,n))
    np.savez('dtm.npz',dtm=dtm,n=n)
