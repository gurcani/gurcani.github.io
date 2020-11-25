#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:07:57 2020

@author: ogurcan
"""
import numpy as np
import matplotlib.pylab as plt
plt.rcParams.update({"text.usetex": True})

def boxed_print(v):
    if(len(v.shape)==1):
        u=v.reshape((v.shape[0],1))
    else:
        u=v
    plt.figure(dpi=100,figsize=(u.shape))
    vm=np.max(np.abs(np.real(u)))
    plt.pcolormesh(u.real.T,zorder=2,edgecolor='k',cmap='bwr',vmin=-vm,vmax=vm,linewidth=2.0)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    for lx in range(u.shape[0]):
        for ly in range(u.shape[1]):
            plt.text(0.5+lx,0.5+ly,'$'+str(u[lx,ly])+'$',horizontalalignment='center',verticalalignment='center',fontsize=14)
    plt.tight_layout()