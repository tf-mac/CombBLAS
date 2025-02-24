/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#pragma once

#include <memory>

#include "SpParHelper.h"
namespace combblas
{

/**
 * The full distribution is actually a two-level distribution that matches the matrix distribution
 * In this scheme, each processor row (except the last) is responsible for t = floor(n/sqrt(p)) elements.
 * The last processor row gets the remaining (n-floor(n/sqrt(p))*(sqrt(p)-1)) elements
 * Within the processor row, each processor (except the last) is responsible for loc = floor(t/sqrt(p)) elements.
 * Example: n=103 and p=16
 * All processors P_ij for i=0,1,2 and j=0,1,2 get floor(floor(102/4)/4) = 6 elements
 * All processors P_i3 for i=0,1,2 get 25-6*3 = 7 elements
 * All processors P_3j for j=0,1,2 get (102-25*3)/4 = 6 elements
 * Processor P_33 gets 27-6*3 = 9 elements
 * Both derived classes, whether sparse or dense, are distributed
 * to processors based on their "length", so that a conversion does not
 * need any communication between sparse and dense formats
 **/
template <class IT, class NT>
class FullyDist
{
   public:
    static_assert(!std::is_same<NT, bool>::value, "Error: NT cannot be bool in FullyDist!");
    explicit FullyDist();
    explicit FullyDist(IT globallen);
    /* ABAB: This clashes with FullyDist(IT globallen) signature on MPICH based systems that #define MPI_Comm to be an
    INT FullyDist( MPI_Comm world):glen(0)
    {
            commGrid.reset(new CommGrid(world, 0, 0));
    }*/
    explicit FullyDist(std::shared_ptr<CommGrid> grid);
    FullyDist(std::shared_ptr<CommGrid> grid, IT globallen);

    FullyDist<IT, NT> &operator=(const FullyDist<IT, NT> &rhs);

    IT LengthUntil() const;
    IT RowLenUntil() const;
    IT RowLenUntil(int k) const;
    IT MyLocLength() const;
    IT MyRowLength() const;
    IT TotalLength() const;
    int Owner(IT gind, IT &lind) const;
    int OwnerWithinRow(IT n_thisrow, IT ind_withinrow, IT &lind) const;

   protected:
    std::shared_ptr<CommGrid> commGrid;
    IT glen;  // global length (actual "length" including zeros)
};

}  // namespace combblas
