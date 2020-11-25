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
c=fftwpp.HConvolution(f.shape)
fc,gc=f.copy(),g.copy()
c.convolve(fc,gc)
h=fc

print("f=",f)
print("g=",g)
print("h=",h)

rfft=np.fft.rfft
irfft=np.fft.irfft

fr=irfft(f,3*m)
gr=irfft(g,3*m)
h2=rfft(fr*gr)[0:m]*3*m

print("h2=",h2)

fex=np.zeros(3*m,dtype=complex)
gex=np.zeros(3*m,dtype=complex)

fex[0:m]=f
gex[0:m]=g
fex[2*m+1:]=np.flipud(f[1:].conj())
gex[2*m+1:]=np.flipud(g[1:].conj())
fex[0]=fex[0].real
gex[0]=gex[0].real

h3=np.zeros_like(g)

for j in range(h3.shape[0]):
    for l in range(3*m):
        h3[j]+=gex[j-l]*fex[l]

print("h3=",h3)

# to see that the result is independent of further padding:

fex=np.zeros(4*m,dtype=complex)
gex=np.zeros(4*m,dtype=complex)

fex[0:m]=f
gex[0:m]=g
fex[3*m+1:]=np.flipud(f[1:].conj())
gex[3*m+1:]=np.flipud(g[1:].conj())
fex[0]=fex[0].real
gex[0]=gex[0].real

h4=np.zeros_like(g)

for j in range(h4.shape[0]):
    for l in range(4*m):
        h4[j]+=gex[j-l]*fex[l]

print("h4=",h4)

fft=np.fft.fft
ifft=np.fft.ifft

h5=fft(ifft(fex)*ifft(gex))[:m]*4*m
print("h5=",h5)