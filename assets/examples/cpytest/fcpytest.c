#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

typedef struct fcpy_pars_{
  int n;
  void (*fn)(complex *, double, int, double *);
} fcpy_pars;

fcpy_pars* init_pars(int n,void (*fn)(complex *, double, int, double *)){
  fcpy_pars *pars=malloc(sizeof(fcpy_pars));
  pars->n=n;
  pars->fn=fn;
  return pars;
}

void fcpytest(fcpy_pars *pars, complex *y, double t, double *res){
  pars->fn(y,t,pars->n,res);
}
