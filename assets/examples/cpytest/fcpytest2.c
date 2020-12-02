#include <complex.h>

void testabssum(complex **F, double *res, int n, int N){
  int j,l;
  for (j=0;j<n;j++){
    res[j]=0;
    for (l=0;l<N;l++){
      res[j]+=cabs(F[l][j]);
    }
  }
}
