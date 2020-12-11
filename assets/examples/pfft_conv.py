#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:56:01 2020

@author: ogurcan
"""
from mpi4py import MPI
from mpi4py_fft import PFFT,newDistArray, DistArray
import numpy as np
from numba import njit

@njit(fastmath=True)
def mult_out62(F):
    for i in range(F.shape[1]):
        for j in range(F.shape[2]):
            dxphi=F[0,i,j];
            dyphi=F[1,i,j];
            dxom=F[2,i,j];
            dyom=F[3,i,j];
            dxn=F[4,i,j];
            dyn=F[5,i,j];
            F[0,i,j]=(dxphi*dyom-dyphi*dxom)
            F[1,i,j]=(dxphi*dyn-dyphi*dxn)

@njit(fastmath=True)
def mult_in(u,F,kx,ky,ksqr):
    for i in range(F.shape[1]):
        for j in range(F.shape[2]):
                F[0,i,j]=1j*kx[i,j]*u[0,i,j]
                F[1,i,j]=1j*ky[i,j]*u[0,i,j]
                F[2,i,j]=-1j*ksqr[i,j]*kx[i,j]*u[0,i,j]
                F[3,i,j]=-1j*ksqr[i,j]*ky[i,j]*u[0,i,j]
                F[4,i,j]=1j*kx[i,j]*u[1,i,j]
                F[5,i,j]=1j*ky[i,j]*u[1,i,j]

def hermitian_symmetrize(g):
    ll=g.local_slice()
    Nx=g.shape[-2]
    if(ll[-1].start==0):
        x0=0
        x1=int(Nx/2)
        ll0=ll[:-2]+(slice(x0,x0+1,None),slice(0,1,None))
        g[ll0]=g[ll0].real
        ll1=ll[:-2]+(slice(x0-1,x1,-1),slice(0,1,None))
        ll2=ll[:-2]+(slice(x0+1,x1,None),slice(0,1,None))
        g[ll1]=g[ll2].conj()
    return g

class hwconv:
    def __init__(self,Nx,Ny,padx=1.5,pady=1.5,num_in=6, num_out=2,
                 fmultin=mult_in, fmultout=mult_out62,comm=MPI.COMM_WORLD):
        self.comm=comm
        self.num_in=num_in
        self.num_out=num_out
        self.nar=max(num_in,num_out)
        self.ffti=PFFT(comm,shape=(self.num_in,Nx,Ny),axes=(1,2), grid=[1,-1,1], padding=[1,1.5,1.5],collapse=False)
        self.ffto=PFFT(comm,shape=(self.num_out,Nx,Ny),axes=(1,2), grid=[1,-1,1], padding=[1,1.5,1.5],collapse=False)
        self.datk=newDistArray(self.ffti,forward_output=True)
        self.dat=newDistArray(self.ffti,forward_output=False)
        lkx=np.r_[0:int(Nx/2),-int(Nx/2):0]
        lky=np.r_[0:int(Ny/2+1)]
        self.kx=DistArray((Nx,int(Ny/2+1)),subcomm=(1,0),dtype=float,alignment=0)
        self.ky=DistArray((Nx,int(Ny/2+1)),subcomm=(1,0),dtype=float,alignment=0)
        self.kx[:],self.ky[:]=np.meshgrid(lkx[self.kx.local_slice()[0]],lky[self.ky.local_slice()[1]],indexing='ij')
        self.ksqr=self.kx**2+self.ky**2
        self.fmultin=fmultin
        self.fmultout=fmultout

    def convolve(self,u):
        hermitian_symmetrize(u)
        if(u.local_slice()[2].stop==u.global_shape[2]):
            u[:,:,-1]=0
        u[:,int(Nx/2),:]=0
        self.fmultin(u,self.datk,self.kx,self.ky,self.ksqr)
        self.ffti.backward(self.datk,self.dat)
        self.fmultout(self.dat)
        self.ffto.forward(self.dat[:self.num_out,],self.datk[:self.num_out,])
        if(self.datk.local_slice()[2].stop==self.datk.global_shape[2]):
            self.datk[:,:,-1]=0
        self.datk[:,int(Nx/2),:]=0
        return self.datk[:self.num_out,]

Nx,Ny=8,8
h=hwconv(Nx,Ny)
uk=DistArray((2,Nx,int(Ny/2+1)),subcomm=(1,1,0),dtype=complex,alignment=1)
vk=DistArray((2,Nx,int(Ny/2+1)),subcomm=(1,1,0),dtype=complex,alignment=1)
kx=DistArray((Nx,int(Ny/2+1)),subcomm=(1,0),dtype=float,alignment=0)
ky=DistArray((Nx,int(Ny/2+1)),subcomm=(1,0),dtype=float,alignment=0)
inds=np.r_[int(Nx/2):Nx,0:int(Nx/2)]
uk[0,:,:]=np.array([[inds[l-1]+1j*m for m in np.r_[uk.local_slice()[2]]] for l in np.r_[uk.local_slice()[1]]])
uk[1,:,:]=np.array([[2*inds[l-1]+1j*(m+1) for m in np.r_[uk.local_slice()[2]]] for l in np.r_[uk.local_slice()[1]]])
vk[:]=h.convolve(uk)
print(np.rint(vk))