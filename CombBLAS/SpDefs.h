/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/


#ifndef _SP_DEFS_H_
#define _SP_DEFS_H_

#include "SequenceHeaps/knheap.C"

#define GRIDMISMATCH 3001
#define DIMMISMATCH 3002

// force 8-bytes alignment in heap allocated memory
#ifndef ALIGN
#define ALIGN 8
#endif

#ifndef THRESHOLD
#define THRESHOLD 5	// if range1.size() / range2.size() < threshold, use scanning based indexing
#endif

#ifndef MEMORYINBYTES
#define MEMORYINBYTES  1048576	// 1 MB, it is advised to define MEMORYINBYTES to be "at most" 1/3 of available memory per core
#endif

#endif
