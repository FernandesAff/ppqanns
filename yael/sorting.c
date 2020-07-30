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

#include <assert.h>
#include <string.h>
#include <math.h>

#include "sorting.h"
#include "machinedeps.h"
#include "binheap.h"
#include "vector.h"

#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))


static int compare_for_k_min (const void *v1, const void *v2)
{
  return *(*(float **) v1) > *(*(float **) v2) ? 1 : -1;
}



/*--------------------------------------------------------------------------
  The following function are related to the Hoare selection algorithm (also know as quickselect). 
  This is a "lazy" version of the qsort algorithm. 
  It is used to find some quantile of the values in a table 
*/

#define PERM(i) (*perm[i])
#define SWAPFPTR(i,j) {const float* tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; }

/* order perm[i0..i1-1] such that *perm[i] <= *perm[j]
   forall i0<=i<q and q<=j<i1  */
static void hoare_selectp (const float **perm, int i0, int i1, int q)
{
  float pivot = PERM(i0);
  int j0, j1, lim;
  assert (i1 - i0 > 1 && q > i0 && q < i1);

  for (j0 = i0, j1 = i1 ; 1 ; ) {
    while (j0 < j1 - 1) {
      j0++;
      if (PERM(j0) > pivot)
        goto endseginf;
    }
    lim = j1;
    break;
  endseginf:
    while (j1 - 1 > j0) {
      j1--;
      if (PERM(j1) <= pivot)
        goto endsegsup;
    }
    lim = j0;
    break;
  endsegsup:
    SWAPFPTR (j0, j1);
  }
  assert (lim > i0);
  if (lim == i1) {
    SWAPFPTR (i0, i1 - 1);
    lim = i1 - 1;
  }
  if (lim == q)
    return; 
  else if (q < lim)
    hoare_selectp (perm, i0, lim, q);
  else
    hoare_selectp (perm, lim, i1, q);
}

#undef PERM
#undef SWAPFPTR


/*  The same Hoare algorithm, but which that modifies its input */
#define PERM(i) f[i]
#define SWAPFLOAT(i,j) {float tmp = f[i]; f[i]=f[j]; f[j]=tmp; }

static void hoare_select_f (float *f, int i0, int i1, int q)
{
  float pivot = PERM(i0);
  int j0, j1, lim;
  assert (i1 - i0 > 1 && q > i0 && q < i1);

  for (j0 = i0, j1 = i1 ; 1 ; ) {
    while (j0 < j1 - 1) {
      j0++;
      if (PERM(j0) > pivot)
        goto endseginf;
    }
    lim = j1;
    break;
  endseginf:
    while (j1 - 1 > j0) {
      j1--;
      if (PERM(j1) <= pivot)
        goto endsegsup;
    }
    lim = j0;
    break;
  endsegsup:
    SWAPFLOAT (j0, j1);
  }
  assert (lim > i0);
  if (lim == i1) {
    SWAPFLOAT (i0, i1 - 1);
    lim = i1 - 1;
  }
  if (lim == q)
    return;                     /* mission accomplished */
  if (q < lim)
    hoare_select_f (f, i0, lim, q);
  else
    hoare_select_f (f, lim, i1, q);
}

#undef PERM
#undef SWAPFLOAT


/*--- Idem for smallest ---*/

/* Hoare version */
static void fvec_k_min_hoare (const float *val, int n, int *idx, int k)
{

  const float **idx_ptr = NEWA (const float *, n);

  int i;
  for (i = 0; i < n; i++)
    idx_ptr[i] = val + i;

  if (k < n)
    hoare_selectp (idx_ptr, 0, n, k);

  /* sort lower part of array */
  qsort (idx_ptr, k, sizeof (*idx_ptr), compare_for_k_min);

  for (i = 0; i < k; i++)
    idx[i] = idx_ptr[i] - val; 

  free (idx_ptr);
}


/* maxheap version */
static void fvec_k_min_maxheap (const float *val, int n,
                                     int *idx, int k)
{
  fbinheap_t *mh = fbinheap_new (k);
  int i;

  for (i = 0; i < n; i++)
    fbinheap_add (mh, i, val[i]);    /* -val because we want maxes instead of mins */

  fbinheap_sort_labels (mh, idx);
  fbinheap_delete (mh);
}


void fvec_k_min (const float *val, int n, int *idx, int k)
{
  assert (k <= n);

  if (n == 0 || k == 0)
    return;

  if (k == 1) {
    *idx = fvec_arg_min (val, n);
    return; 
  }

  /* TODO: find out where the limit really is */
  if (n > 3 * k)
    fvec_k_min_maxheap (val, n, idx, k); 
  else
    fvec_k_min_hoare (val, n, idx, k);
}



/*********************************************************************
 * Simple functions 
 *********************************************************************/

#ifdef HAVE_TLS

static __thread const float * tab_to_sort_f;

static int compare_for_sort_index_f (const void *v1, const void *v2)
{
#elif defined(HAVE_QSORT_R)
static int compare_for_sort_index_f (void *thunk, const void *v1, const void *v2)
{
  const float *tab_to_sort_f=thunk;
#else 
#error "please provide some kind of thread-local storage"
#endif
  
  float dt = tab_to_sort_f[*(int *)v1] - tab_to_sort_f[*(int *)v2];
  if (dt) 
    return dt>0 ? 1 : -1;
  return *(int *)v1 - *(int *)v2;
}



void fvec_sort_index(const float *tab,int n,int *perm) {
  int i;

  for (i = 0 ; i < n ; i++) 
    perm[i] = i;

#ifdef HAVE_TLS
  tab_to_sort_f = tab;
  qsort (perm, n, sizeof(int), compare_for_sort_index_f);
#elif defined(HAVE_QSORT_R)
  qsort_r (perm, n, sizeof(int), (void*)tab, compare_for_sort_index_f);
#endif
}



#ifdef HAVE_TLS

static __thread const int * tab_to_sort;

static int compare_for_sort_index (const void *v1, const void *v2)
{
#elif defined(HAVE_QSORT_R)
static int compare_for_sort_index (void *thunk, const void *v1, const void *v2)
{
  const int *tab_to_sort = thunk;
#else 
#error "please provide some kind of thread-local storage"
#endif
  
  int dt = tab_to_sort[*(int *)v1] - tab_to_sort[*(int *)v2];
  if (dt) 
    return dt;
  return *(int *)v1 - *(int *)v2;
}


void ivec_sort_index (const int *tab, int n, int *perm) 
{
  int i;

  for (i = 0 ; i < n ; i++) 
    perm[i] = i;

#ifdef HAVE_TLS
  tab_to_sort = tab;
  qsort (perm, n, sizeof(int), compare_for_sort_index);
#elif defined(HAVE_QSORT_R)
  qsort_r (perm, n, sizeof(int), (void*)tab, compare_for_sort_index);
#endif
}


float fvec_median (float *f, int n)
{
  if(n == 0) 
    return 0.0 / 0.0; 

  if (n == 1) 
    return f[0];

  int halfn = n / 2;
  int j;

  hoare_select_f (f, 0, n, halfn);

  float min_upper = f[halfn];
  for (j = halfn + 1; j < n; j++)
    if (f[j] < min_upper)
      min_upper = f[j];

  if (n % 2 == 1)
    return min_upper;
  else {
    float max_lower = f[0];
    for (j = 1; j < halfn; j++)
      if (f[j] > max_lower)
        max_lower = f[j];
    return 0.5 * (min_upper + max_lower);
  }
}


int fvec_arg_max (const float *f, long n) 
{
  assert (n > 0);
  float m = f[0];
  long i,i0 = 0;
  for (i = 1 ; i < n ; i++) 
    if (f[i] > m) {
      m = f[i]; 
      i0 = i; 
    }
  return i0;
}


int fvec_arg_min (const float *f, long n) 
{
  assert (n > 0);
  float m = f[0];
  long i, i0 = 0;
  for (i = 1 ; i < n ; i++) 
    if(f[i] < m) {
      m = f[i]; 
      i0 = i; 
    }
  return i0;
}