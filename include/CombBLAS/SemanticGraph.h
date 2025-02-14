

#ifndef _SP_SEMANTIC_GRAPH_H_
#define _SP_SEMANTIC_GRAPH_H_

#include <mpi.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
// TR1 includes belong in CombBLAS.h

#include "CommGrid.h"
#include "Deleter.h"
#include "Friends.h"
#include "FullyDistVec.h"
#include "LocArr.h"
#include "MPIType.h"
#include "Operations.h"
#include "SpDCCols.h"
#include "SpDefs.h"
#include "SpHelper.h"
#include "SpMat.h"
#include "SpParHelper.h"
#include "SpTuples.h"

namespace combblas
{

template <class IT, class NT, class DER>
class SemanticGraph
{
   public:
    SemanticGraph(IT total_m, IT total_n, const FullyDistVec<IT, IT>&, const FullyDistVec<IT, IT>&, const FullyDistVec<IT, NT>&);  // matlab sparse
    typename typedef SpParMat<IT, NT, SpDCCols<IT, NT> > PSpMat;  // TODO: Convert to 32-bit local indices
    typename typedef FullyDistVec<IT, NT> PVec;

   private:
    PSpMat SemMat;
    PVec SemVec;
}

}  // namespace combblas

#endif
