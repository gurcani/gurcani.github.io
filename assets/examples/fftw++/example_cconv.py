#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 21:09:54 2020

@author: ogurcan
"""
import fftwpp
import numpy as np
m=8

f=np.array([l+1j*(l+1) for l in range(m)])
g=np.array([l+1j*(2*l+1) for l in range(m)])
c=fftwpp.Convolution(f.shape)
fc,gc=f.copy(),g.copy()
c.convolve(fc,gc)
h=fc

print("f=",f)
print("g=",g)
print("h=",h)

h2=np.zeros_like(g)

for j in range(m):
    for l in range(j+1):
        h2[j]+=g[j-l]*f[l]

print("h2=",h2)