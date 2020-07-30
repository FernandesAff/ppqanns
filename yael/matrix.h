#ifndef __matrix_h
#define __matrix_h

/*---------------------------------------------------------------------------*/
/* Standard operations                                                       */
/*---------------------------------------------------------------------------*/

/*! Allocate a new nrow x ncol matrix */
float *fmat_new (int nrow, int ncol);

void fmat_mul_full(const float *left, const float *right,
                   int m, int n, int k,
                   const char *transp,
                   float *result);

/*! same as fmat_mul_full, matrices may be non-packed (yes, this is close to sgemm) */
void fmat_mul_full_nonpacked(const float *left, const float *right,
                             int m, int n, int k,
                             const char *transp,
                             int ld_left, int ld_right, 
                             float *result,
                             int ld_result);

/*! same as fmat_mul_full, right(n,k) transposed */
void fmat_mul_tr (const float *left, const float *right, int m, int n, int k, float *mout);

/*! display the matrix in matlab-parsable format */
void fmat_print (const float *a, int nrow, int ncol);

#endif 