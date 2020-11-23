Importing fftwpp, intializing and displaying f ang g.


```python
import fftwpp
import numpy as np
m=8
f=np.array([l+1j*(l+1) for l in range(m)])
g=np.array([l+1j*(2*l+1) for l in range(m)])
print("f=",f)
print("g=",g)
```

    f= [0.+1.j 1.+2.j 2.+3.j 3.+4.j 4.+5.j 5.+6.j 6.+7.j 7.+8.j]
    g= [0. +1.j 1. +3.j 2. +5.j 3. +7.j 4. +9.j 5.+11.j 6.+13.j 7.+15.j]


Initializing and Computing the convolutiong using fftw++ and displaying the result


```python
c=fftwpp.Convolution(f.shape)
c.convolve(f,g)
h=f.copy()
print("h=",h)
```

    h= [  -1.  +0.j   -5.  +2.j  -13.  +9.j  -26. +24.j  -45. +50.j  -71. +90.j
     -105.+147.j -148.+224.j]


reinitializing f and g as before and computing the convolution by hand for comparison


```python
f=np.array([l+1j*(l+1) for l in range(m)])
g=np.array([l+1j*(2*l+1) for l in range(m)])
h2=np.zeros_like(g)
for j in range(m):
    for l in range(j+1):
        h2[j]+=g[j-l]*f[l]
print("h2=",h2)
```

    h2= [  -1.  +0.j   -5.  +2.j  -13.  +9.j  -26. +24.j  -45. +50.j  -71. +90.j
     -105.+147.j -148.+224.j]

