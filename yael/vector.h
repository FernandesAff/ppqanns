/*
Copyright © INRIA 2009-2014.
Authors: Matthijs Douze & Herve Jegou
Contact: matthijs.douze@inria.fr  herve.jegou@inria.fr

This file is part of Yael.

    Yael is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Yael is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Yael.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __vector_h
#define __vector_h

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <errno.h>

/*-------------- Basic math operations ----------------*/


/*! generate a random sample, mean 0 variance 1 */
double gaussrand ();


/*---------------------------------------------------------------------------*/
/*! @addtogroup vector
 *  @{
 */



/*! @defgroup vector
 * Vectors are represented as C arrays of basic elements. Functions
 * operating on them are prefixed with:
 *
 * ivec_: basic type is int
 *
 * fvec_: basic type is float
 *
 * Vector sizes are passed explicitly, as long int's to allow for
 * large arrays on 64 bit machines. Vectors can be free'd with free().
 *
 *
 * Arrays of vectors are stored contiguously in memory. An array of n
 * float vectors of dimension d is
 *
 *   float *fv
 *
 * The i'th element of vector j of vector array vf, where 0 <= i < d
 * and 0 <= j < n is
 *
 *   vf[ j * d + i ]
 *
 * It can also be seen as a column-major matrix of size d, n.
 *
 */


/*! Alloc a new aligned vector of floating point values -- to be
 *  de-allocated with free. Some operations may be faster if input
 *  arrays are allocated with this function (data is suitably
 *  aligned). */
float * fvec_new (long n);

/*! Alloc an int array -- to be de-allocated with free. */
int *ivec_new (long n);

/*! create a vector initialized with 0's. */
float * fvec_new_0 (long n);

/*! create a vector initialized with 0's. */
int *ivec_new_0 (long n);

/*!  create a vector initialized with a specified value. */
float *fvec_new_set (long n, float val);

/*!  create a vector initialized with a specified value. */
int *ivec_new_set (long n, int val);

/*!  same as fvec_randn, with seed for thread-safety */
void fvec_randn_r (float * v, long n, unsigned int seed);

/*!  new vector initialized with another vector */
float * fvec_new_cpy (const float * v, long n);

/*!  select k random indexes among n (without repetition) */
int *ivec_new_random_idx  (int n, int k);

/*!  same as ivec_new_random_perm, thread-safe, with a random seed */
int *ivec_new_random_perm_r (int n, unsigned int seed);

/*!  same as ivec_new_random_idx, thread-safe with a random seed  */
int *ivec_new_random_idx_r  (int n, int k, unsigned int seed);

/*! resize a vector (realloc). Usage: v = fvec_resize (v, n). */
float * fvec_resize (float * v, long n);

/*! resize a vector (realloc). Usage: v = fvec_resize (v, n). */
int * ivec_resize (int * v, long n);

/*!  count occurrences
   @param k is the range of the values that may be encountered (assuming start at 0). Values outside the range trigger an assertion!
   @param v is the vector of values to be histrogramized, of length n
*/
int * ivec_new_histogram (int k, const int * v, long n);

/*!  count occurences of a value in the vector */
long fvec_count_occurrences (const float * v, long n, float val);
long ivec_count_occurrences(const int * v, long n, int val);

/*---------------------------------------------------------------------------*/
/* Input/Output functions                                                    */
/* I/O of a single vector is supported only if it is smaller than 2^31       */
/*---------------------------------------------------------------------------*/

static long xvecs_fsize(long unitsize, const char * fname, int *d_out, int *n_out);
long bvecs_fsize (const char * fname, int *d_out, int *n_out);

/*!  write a vector into an open file */
int ivec_fwrite(FILE *f, const int *v, int d);
int fvec_fwrite(FILE *f, const float *v, int d);

/*!  load float vector without allocating memory
 *
 * Fills n*d array with as much vectors read from fname as possible.
 * Returns nb of vectors read, or <0 on error.
 */
int fvecs_read (const char *fname, int d, int n, float *v);

int b2fvecs_read (const char *fname, int d, int n, float *v);

/*!  load float vectors from an open file. Return the dimension */
int fvec_fread (FILE * f, float * v, int d_alloc);

/*!  read an integer vector file from an open file and return the dimension */
int ivec_fread (FILE *f, int * v, int d_alloc);

long b2fvecs_fread (FILE * f, float * v, long n);

int b2fvec_fread (FILE * f, float * v);

/*!  display a float vector */
void fvec_print (const float * v, int n);

/*!  display an integer vector */
void ivec_print (const int * v, int n);




/*---------------------------------------------------------------------------*/
/* Vector manipulation and elementary operations                             */
/*---------------------------------------------------------------------------*/

/*!  Set all the components of the vector v to 0 */
void fvec_0 (float * v, long n);
void ivec_0 (int * v, long n);

/*!  Set all the components of the vector v to the value val */
void fvec_set (float * v, long n, float val);

/*!  copy the vector from v2 to v1 */
void ivec_cpy (int * vdest, const int * vsource, long n);
void fvec_cpy (float * vdest, const float * vsource, long n);

/*!  Multiply or divide a vector by a scalar */
void fvec_mul_by (float * v, long n, double scal);
void fvec_div_by (float * v, long n, double scal);

/*!  Add or subtract two vectors. The result is stored in v1. */
void fvec_add (float * v1, const float * v2, long n);
void fvec_sub (float * v1, const float * v2, long n);

/*!  Normalize the vector for the given Minkowski norm.
  The function return the norm of the original vector.
  If the vector is all 0, it will be filled with NaNs.
  This case can be identified when the return value is 0.
  Infinty norm can be obtained with norm=-1 */
double fvec_normalize (float * v, long n, double norm);

void fvec_sqr (float * v, long n);

/*---------------------------------------------------------------------------*/
/* Vector measures and statistics                                            */
/*---------------------------------------------------------------------------*/

/*!  compute the sum of the value of the vector */
double fvec_sum (const float * v, long n);

/*! cumulative sum */
void fvec_cumsum(float * v, long n);

/*!  sum of squared components */
double fvec_sum_sqr (const float * v, long n);

/*!  compute the norm of a given vector (norm=-1 => infinty norm) */
double fvec_norm (const float * v, long n, double norm);

/*!  compute squared norm 2 */
double fvec_norm2sqr (const float * v, long n);

double ivec_unbalanced_factor(const int *hist, long n);

/*---------------------------------------------------------------------------*/
/* Distances and similarities                                                */
/*---------------------------------------------------------------------------*/

/*!  Return the square L2 distance between vectors */
double fvec_distance_L2sqr (const float * v1, const float * v2, long n);

/*---------------------------------------------------------------------------*/
/* Elaborate vector manipulations                                            */
/*---------------------------------------------------------------------------*/

void fvec_cpy_subvectors (const float * v, int * idx, int d, int nout, float * vout);

void b2fvec_cpy_subvectors (const unsigned char * v, int * idx, int d, int nout, float * vout);

#endif
