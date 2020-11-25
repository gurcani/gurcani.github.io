#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 21:09:54 2020

@author: ogurcan
"""
import fftwpp
import numpy as np
mx,my=4,4
nx=2*mx-1
nyc=2*my-1

def hermitian_symmetrize(f,x0=-1):
    g=f.copy()
    if (x0==-1) : 
        x0=int(g.shape[0]/2)
    g[x0,0]=g[x0,0].real
    for l in range(1,x0+1):
        g[x0-l,0]=g[x0+l,0].conj()
    return g

f=np.array([[l+1j*m for l in range(nx)] for m in range(my)]).T
g=np.array([[2*l+1j*(m+1) for l in range(nx)] for m in range(my)]).T

c=fftwpp.HConvolution(f.shape)
fc,gc=f.copy(),g.copy()
c.convolve(fc,gc)
h=fc

print("f=",f)
print("g=",g)
h=np.rint(h.real)+1j*np.rint(h.imag)
print("h=",h)

gsym=hermitian_symmetrize(g)
fsym=hermitian_symmetrize(f)

nx_p=int(np.ceil(nx*3/2))
my_p=int(np.ceil(my*3/2))

gsc=np.zeros((nx_p,my_p),dtype=complex)
fsc=np.zeros((nx_p,my_p),dtype=complex)

gsc[np.r_[:mx,nx_p-mx+1:nx_p],:my]=gsym[np.r_[mx-1:nx,:mx-1],:]
fsc[np.r_[:mx,nx_p-mx+1:nx_p],:my]=fsym[np.r_[mx-1:nx,:mx-1],:]

rfft2=np.fft.rfft2
irfft2=np.fft.irfft2

fr=irfft2(fsc)
gr=irfft2(gsc)

h2=rfft2(fr*gr)*np.prod(fr.shape)
h2=(np.rint(h2.real)+1j*np.rint(h2.imag))[np.r_[-mx+1:mx],:my]
print('h2=',h2)

fk=np.zeros((nx_p,2*my_p-1),dtype=complex)
gk=np.zeros((nx_p,2*my_p-1),dtype=complex)

fk[:,:my_p]=fsc
gk[:,:my_p]=gsc
fk[0,-my_p+1:]=np.flipud(fsc[0,1:]).conj()
gk[0,-my_p+1:]=np.flipud(gsc[0,1:]).conj()
fk[1:,-my_p+1:]=np.flipud(np.fliplr(fsc[1:,1:])).conj()
gk[1:,-my_p+1:]=np.flipud(np.fliplr(gsc[1:,1:])).conj()

fft2=np.fft.fft2
ifft2=np.fft.ifft2
fr=ifft2(fk)
gr=ifft2(gk)
h3=fft2(fr*gr)*np.prod(gr.shape)
h3=(np.rint(h3.real)+1j*np.rint(h3.imag))[np.r_[-mx+1:mx],:my]
print("h3=",h3)

h4=np.zeros_like(fk)

for jx in range(h4.shape[0]):
    for jy in range(h4.shape[1]):
        for lx in range(h4.shape[0]):
            for ly in range(h4.shape[1]):
                h4[jx,jy]+=gk[jx-lx,jy-ly]*fk[lx,ly]
h4=(np.rint(h4.real)+1j*np.rint(h4.imag))[np.r_[-mx+1:mx],:my]
print("h4=",h4)
