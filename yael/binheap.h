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

#ifndef __binheap_h
#define __binheap_h

#include <stdlib.h>

/*---------------------------------------------------------------------------*/
/*! @addtogroup binheap
 *  @{
 */


/*! @defgroup binheap
  This structure is used, in particular, to find the maxk smallest
  elements of a possibly unsized stream of values. 
*/

/*! Binary heap used as a maxheap. 
  Element (label[1],val[1]) always contains the maximum value of the binheap. 
*/
struct fbinheap_s {
  float * val;     /*!< valid values are val[1] to val[k] */
  int * label;     /*!< idem for labels */
  int k;           /*!< number of elements stored  */
  int maxk;        /*!< maximum number of elements */
};

typedef struct fbinheap_s fbinheap_t;

/*! return the size of a maxheap structure 
 @param maxk the maximum number of elements that the structure will receive */
size_t fbinheap_sizeof (int maxk); 

/*! A binheap can be stored in an externally allocated memory area 
  of fbinheap_sizeof(maxk) bytes. The fbinheap_init() function is used 
  to initialize this memory area */
void fbinheap_init (fbinheap_t *bh, int maxk);

/*! free allocated memory */
void fbinheap_delete (fbinheap_t * bh);

/*! insert an element on the heap (if the value val is small enough) */
void fbinheap_add (fbinheap_t * bh, int label, float val);

/*! remove largest value from binheap (low-level access!) */
void fbinheap_pop (fbinheap_t * bh);

/*! add n elements on the heap, using the set of labels starting at label0  */
void fbinheap_addn_label_range (fbinheap_t * bh, int n, int label0, const float * v);

/*! output the labels in increasing order of associated values 
  @param bh the maxheap structure
  @pram perm the array that receive the output permutation order (pre-allocated)
*/

/*! output both sorted results: labels and corresponding values  */
void fbinheap_sort (fbinheap_t * bh, int * labels, float *v);

#endif