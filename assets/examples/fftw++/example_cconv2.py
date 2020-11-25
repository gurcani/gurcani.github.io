#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 21:09:54 2020

@author: ogurcan
"""
import fftwpp
import numpy as np
mx,my=4,4

f=np.array([[l+1j*m for l in range(mx)] for m in range(my)]).T
g=np.array([[2*l+1j*(m+1) for l in range(mx)] for m in range(my)]).T

c=fftwpp.Convolution(f.shape)
fc,gc=f.copy(),g.copy()
c.convolve(fc,gc)
h=fc

print("f=",f)
print("g=",g)
print("h=",np.rint(h.real)+1j*np.rint(h.imag))


h2=np.zeros_like(g)

for jx in range(mx):
    for jy in range(my):
     for lx in range(jx+1):
         for ly in range(jy+1):
             h2[jx,jy]+=g[jx-lx,jy-ly]*f[lx,ly]

print("h2=",h2)