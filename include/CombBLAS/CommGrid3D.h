#ifndef COMBBLAS_COMMGRID3D_H
#define COMBBLAS_COMMGRID3D_H

#include <mpi.h>

#include <memory>

#include "CommGrid.h"
#include "SpDefs.h"

namespace combblas
{

class CommGrid3D
{
   public:
    // Constructor: create a 3D grid using the provided MPI communicator, number of layers,
    // number of processors per row and per column in each layer, and a flag for special distribution.
    CommGrid3D(MPI_Comm world, int nlayers, int nrowproc, int ncolproc, bool special = false);

    // Destructor: frees MPI communicators.
    ~CommGrid3D();

    // Return the global rank of a processor based on its layer, row, and column in the 3D grid.
    int GetRank(int layerrank, int rowrank, int colrank);

    int GetGridLayers();
    int GetGridRows();
    int GetGridCols();
    int GetSize();
    bool isSpecial();

    MPI_Comm& GetWorld();
    MPI_Comm& GetFiberWorld();
    MPI_Comm& GetLayerWorld();
    std::shared_ptr<CommGrid> GetCommGridLayer();

    int GetRankInWorld();
    int GetRankInFiber();
    int GetRankInLayer();

   private:
    bool special;
    int gridRows;    // Processors along a row in each layer
    int gridCols;    // Processors along a column in each layer
    int gridLayers;  // Number of layers in the 3D grid
    int myrank;      // Rank in the global communicator
    int rankInFiber;
    int rankInLayer;
    MPI_Comm world3D;
    MPI_Comm layerWorld;
    MPI_Comm fiberWorld;
    std::shared_ptr<CommGrid> commGridLayer;  // 2D grid for the layer that this processor belongs to
};

}  // namespace combblas

#endif  // COMBBLAS_COMM_GRID_3D_H