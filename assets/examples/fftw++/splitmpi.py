#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:28:19 2020

@author: ogurcan
"""
def splitmpi(N,rank,size):
    nperpe = int(N/size)
    nrem = N - size*nperpe
    local_N = nperpe+(rank < nrem)
    loc_base = rank*nperpe+min(rank,nrem)
    return local_N,loc_base


def ceilquotient(a,b):
    return int((a+b-1)/b)

def localdimension(N, rank, size):
    n=ceilquotient(N,size)
    start=n*rank
    extra=N-start
    if(extra < 0):
        extra=0
    if(n > extra):
        n=extra
    return n,start