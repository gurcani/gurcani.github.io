#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 03:16:27 2020

@author: ogurcan
"""
import numpy as np
import matplotlib.pylab as plt
flname='dtm.npz'
fl=np.load(flname)
n=fl['n']
dtm=fl['dtm']
labels=['einsum','multiply','numba','C']
width=0.2
x=np.arange(len(n))
fig, ax = plt.subplots()
for l in range(len(labels)):
    ax.bar(x-width+l*width*1.1,dtm[:,l,0],width, yerr=dtm[:,l,1],label=labels[l])
ax.set_ylabel('Time in secs')
ax.set_title('number of processes')
ax.set_xticks(x)
ax.set_xticklabels(n)
ax.legend()