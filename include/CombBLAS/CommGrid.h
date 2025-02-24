/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.5 -------------------------------------------------*/
/* date: 10/09/2015 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2015, The Regents of the University of California

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

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "MPIType.h"

namespace combblas
{

class CommGrid
{
   public:
    CommGrid(MPI_Comm world, int nrowproc, int ncolproc);
    ~CommGrid();
    // copy constructor
    CommGrid(const CommGrid& rhs);
    // assignment operator
    CommGrid& operator=(const CommGrid& rhs);
    void CreateDiagWorld();
    bool operator==(const CommGrid& rhs) const;
    inline bool operator!=(const CommGrid& rhs) const { return (!(*this == rhs)); }
    [[nodiscard]] inline bool OnSameProcCol(int rhsrank) const { return (myproccol == ((int)(rhsrank % grcols))); }
    [[nodiscard]] inline bool OnSameProcRow(int rhsrank) const { return (myprocrow == ((int)(rhsrank / grcols))); }
    [[nodiscard]] inline int GetRank(int rowrank, int colrank) const { return rowrank * grcols + colrank; }
    [[nodiscard]] inline int GetRank(int diagrank) const { return diagrank * grcols + diagrank; }
    [[nodiscard]] inline int GetRank() const { return myrank; }
    [[nodiscard]] inline int GetRankInProcRow() const { return myproccol; }
    [[nodiscard]] inline int GetRankInProcCol() const { return myprocrow; }
    [[nodiscard]] inline int GetDiagRank() const
    {
        int rank;
        MPI_Comm_rank(diagWorld, &rank);
        return rank;
    }

    [[nodiscard]] inline int GetRankInProcRow(int wholerank) const { return ((int)(wholerank % grcols)); }
    [[nodiscard]] inline int GetRankInProcCol(int wholerank) const { return ((int)(wholerank / grcols)); }
    [[nodiscard]] inline int GetDiagOfProcRow() const { return myprocrow; }
    [[nodiscard]] inline int GetDiagOfProcCol() const { return myproccol; }
    // For P(i,j), get rank of P(j,i)
    [[nodiscard]] inline int GetComplementRank() const { return ((grcols * myproccol) + myprocrow); }

    MPI_Comm& GetWorld() { return commWorld; }
    MPI_Comm& GetRowWorld() { return rowWorld; }
    MPI_Comm& GetColWorld() { return colWorld; }
    MPI_Comm& GetDiagWorld() { return diagWorld; }
    [[nodiscard]] MPI_Comm GetWorld() const { return commWorld; }
    [[nodiscard]] MPI_Comm GetRowWorld() const { return rowWorld; }
    [[nodiscard]] MPI_Comm GetColWorld() const { return colWorld; }
    [[nodiscard]] MPI_Comm GetDiagWorld() const { return diagWorld; }
    [[nodiscard]] int GetGridRows() const { return grrows; }
    [[nodiscard]] int GetGridCols() const { return grcols; }
    [[nodiscard]] int GetSize() const { return grrows * grcols; }
    [[nodiscard]] int GetDiagSize() const
    {
        int size;
        MPI_Comm_size(diagWorld, &size);
        return size;
    }

    void OpenDebugFile(std::string prefix, std::ofstream& output) const;
    // clang-format off
    friend std::shared_ptr<CommGrid>
    ProductGrid(CommGrid* gridA, CommGrid* gridB, int& innerdim, int& Aoffset, int& Boffset);
    // clang-format on
   private:
    // A "normal" MPI-1 communicator is an intracommunicator; MPI::COMM_WORLD is also an MPI::Intracomm object
    MPI_Comm commWorld, rowWorld, colWorld, diagWorld;

    // Processor grid is (grrow X grcol)
    int grrows, grcols;
    int myprocrow;
    int myproccol;
    int myrank;

    template <class IT, class NT, class DER>
    friend class SpParMat;

    template <class IT, class NT>
    friend class FullyDistSpVec;
};

}  // namespace combblas
