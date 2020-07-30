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

#ifndef SORTING_H_INCLUDED
#define SORTING_H_INCLUDED

/*! @addtogroup sorting
 *  @{  */

/*! @defgroup: sorting  
 *
 * Various sorting functions + a few simple array functions that can
 * be called from python efficiently */

/*! Find the minimum elements of an array.
 * See find_k_max. 
 */
void fvec_k_min(const float *v, int n, int *mins, int k);

/*! return the position of the smallest element of a vector.
  First position in case of ties, n should be >0. */
int fvec_arg_min (const float *f, long n);

/*! return the position of the largest elements of a vector.
  First position in case of ties, n should be >0. */
int fvec_arg_max (const float *f, long n);


/*! computes the median of a float array. Array modified on output! */
float fvec_median (float *f, int n);

/*! return permutation to sort an array. 
 *
 * @param tab(n)    table to sort
 * @param perm(n)   output permutation that sorts table
 * 
 * On output,  
 *
 *      tab[perm[0]] <= tab[perm[1]] <= ... <= tab[perm[n-1]]
 *
 * Is stable. 
 */
void ivec_sort_index (const int *tab, int n, int *perm);

/*! return permutation to sort an array. See ivec_sort_index. */
void fvec_sort_index (const float *tab, int n, int *perm);

/*! Apply a permutation to a vector. The permutation is 
 * typically generated using the ivec_sort_index function. In that 
 *  case the function outputs a sorted array. 
 */

#endif
