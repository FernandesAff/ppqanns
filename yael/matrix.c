#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "vector.h"
#include "matrix.h"
#include "sorting.h"
#include "machinedeps.h"
#include "eigs.h"

#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))
#define NEWAC(type,n) (type*)calloc(sizeof(type),(n))
#define NEW(type) NEWA(type,1)


/* blas/lapack subroutines */

#define real float
#define integer FINTEGER

int sgemm_ (char *transa, char *transb, integer * m, integer *
            n, integer * k, real * alpha, const real * a, integer * lda,
            const real * b, integer * ldb, real * beta, real * c__,
            integer * ldc);

int ssyev_ (char *jobz, char *uplo, integer * n, real * a,
            integer * lda, real * w, real * work, integer * lwork,
            integer * info);


int sgeqrf_ (integer * m, integer * n, real * a, integer * lda,
             real * tau, real * work, integer * lwork, integer * info);

int slarft_ (char *direct, char *storev, integer * n, integer *
             k, real * v, integer * ldv, real * tau, real * t, integer * ldt);

int slarfb_ (char *side, char *trans, char *direct, char *storev, integer * m,
             integer * n, integer * k, real * v, integer * ldv, real * t,
             integer * ldt, real * c__, integer * ldc, real * work,
             integer * ldwork);

int ssyrk_(char *uplo, char *trans, integer *n, integer *k, 
           real *alpha, real *a, integer *lda, real *beta, real *c__, integer *
           ldc);


void sgemv_(const char *trans, integer *m, integer *n, real *alpha, 
                   const real *a, integer *lda, const real *x, integer *incx, real *beta, real *y, 
                   integer *incy);


int sgels_(char *trans, integer *m, integer *n, integer *
           nrhs, float *a, integer *lda, float *b, integer *ldb,
           float *work, integer *lwork, integer *info);


#undef real
#undef integer

/*---------------------------------------------------------------------------*/
/* Standard operations                                                       */
/*---------------------------------------------------------------------------*/

float *fmat_new (int nrow, int ncol)
{
  float *m = fvec_new (nrow * (long)ncol);
  return m;
}

void fmat_mul_full(const float *left, const float *right,
                   int m, int n, int k,
                   const char *transp,
                   float *result) {

  fmat_mul_full_nonpacked(left, right, m, n, k, transp, 
                          (transp[0] == 'N' ? m : k), 
                          (transp[1] == 'N' ? k : n), 
                          result, m);
                       
  
}

void fmat_mul_full_nonpacked(const float *left, const float *right,
                             int mi, int ni, int ki,
                             const char *transp,
                             int ld_left, int ld_right, 
                             float *result,
                             int ld_result) {

  float alpha = 1;
  float beta = 0;
  FINTEGER m=mi,n=ni,k=ki;
  FINTEGER lda = ld_left;
  FINTEGER ldb = ld_right;
  FINTEGER ldc = ld_result; 
  
  sgemm_ ((char*)transp, (char*)(transp+1), &m, &n, &k,
          &alpha, left, &lda, right, &ldb, &beta, result, &ldc);

}

void fmat_mul_tr (const float *left, const float *right, int m, int n, int k, float *mout) {
  fmat_mul_full(left,right,m,n,k,"NT",mout);
}

void fmat_print (const float *a, int nrow, int ncol)
{
  int i, j;

  printf ("[");
  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++)
      printf ("%.5g ", a[i + nrow * j]);
    if (i == nrow - 1)
      printf ("]\n");
    else
      printf (";\n");
  }
}