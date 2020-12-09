#include <complex.h>

void multc(complex *u, complex *a, complex *b, complex *res, unsigned int N){
  int l,j;
  for (j=0;j<2;j++){
    for (l=0;l<N;l++){
      res[N*j+l]=a[N*2*j+N*0+l]*u[N*0+l]+a[N*2*j+N*1+l]*u[N*1+l]+b[N*j+l];
    }
  }
}
