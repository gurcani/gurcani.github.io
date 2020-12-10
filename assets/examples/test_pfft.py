#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:44:51 2020

@author: ogurcan
"""
from mpi4py import MPI
from mpi4py_fft import PFFT,newDistArray
import numpy as np
import matplotlib.pylab as plt

howmany=6
Nx,Ny=128,128
padx,pady=3/2,3/2
Npx,Npy=int(128*padx),int(128*pady)
comm=MPI.COMM_WORLD

pf=PFFT(comm,shape=(howmany,Nx,Ny),axes=(1,2), grid=[1,-1,1], padding=[1,1.5,1.5],collapse=False)
u=newDistArray(pf,forward_output=False)
uk=newDistArray(pf,forward_output=True)

n,x,y=np.meshgrid(np.arange(0,howmany),np.linspace(-1,1,Npx),np.linspace(-1,1,Npy),indexing='ij')
nl,xl,yl=n[u.local_slice()],x[u.local_slice()],y[u.local_slice()]
u[:]=np.sin(4*np.pi*(xl+2*yl))*np.exp(-xl**2/2/0.04-yl**2/2/0.08)*(nl-3)
u0=u.copy()

pf.forward(u,uk)
pf.backward(uk,u)

plt.figure()
plt.pcolormesh(xl[0,].T,yl[0,].T,u0[0,].T-u[0,].T,cmap='twilight_shifted',rasterized=True)
plt.colorbar()
plt.axis('square')
plt.axis([-1,1,-1,1])

u1=u.copy()

pf.forward(u,uk)
pf.backward(uk,u)

plt.figure()
plt.pcolormesh(xl[0,].T,yl[0,].T,u1[0,].T-u[0,].T,cmap='twilight_shifted',rasterized=True)
plt.colorbar()
plt.axis('square')
plt.axis([-1,1,-1,1])

# plt.pause(5)
# plt.figure()
#print('rank=',comm.rank,uk.shape)
# lkx,lky=np.arange(0,Nx),np.arange(0,int(Ny/2+1))
# l=uk.local_slice()[2]
# print('rank=',comm.rank,lky[l])
# plt.pcolormesh(lkx,lky[l],np.log10(np.abs(uk[0,])).T,cmap='twilight_shifted',rasterized=True,vmin=-10,vmax=0)
# plt.axis('square')
# plt.axis([lkx[0],lkx[-1],lky[0],lky[-1]])
# plt.show()
#plt.savefig('pfft_exout_'+str(comm.rank)+'.svg')
# plt.figure()
# i=np.unravel_index(np.argmax(u),u.shape)
# plt.plot(u[:,i[1],i[2]])
# plt.figure()
# i=np.unravel_index(np.argmax(uk),uk.shape)
# plt.plot(uk[:,i[1],i[2]])
