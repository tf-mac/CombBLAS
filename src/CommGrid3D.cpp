

#include "CombBLAS/CommGrid3D.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace combblas
{

CommGrid3D::CommGrid3D(MPI_Comm world, int nlayers, int nrowproc, int ncolproc, bool special)
    : gridLayers(nlayers), gridRows(nrowproc), gridCols(ncolproc), special(special)
{
    int nproc;
    MPI_Comm_dup(world, &world3D);
    MPI_Comm_rank(world3D, &myrank);
    MPI_Comm_size(world3D, &nproc);

    if (nlayers < 1) {
        std::cerr << "A 3D grid can not be created with less than one layer" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, NOTSQUARE);
    }
    if (nproc % nlayers != 0) {
        std::cerr << "Number of processes is not divisible by number of layers" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, NOTSQUARE);
    }
    if (special) {
        if (((int)std::sqrt((float)nlayers) * (int)std::sqrt((float)nlayers)) != nlayers) {
            std::cerr << "Number of layers is not a square number" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, NOTSQUARE);
        }
    }

    int procPerLayer = nproc / nlayers;
    // If no gridRows and gridCols were provided, compute them as the square root of processors per layer.
    if (gridRows == 0 && gridCols == 0) {
        gridRows = (int)std::sqrt((float)procPerLayer);
        gridCols = gridRows;
        if (gridRows * gridCols != procPerLayer) {
            std::cerr << "This version of the Combinatorial BLAS only works on a square logical processor grid in a layer of the 3D grid"
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, NOTSQUARE);
        }
    }
    // Ensure that the total number of processors matches the 3D grid dimensions.
    assert(nproc == (gridRows * gridCols * gridLayers));

    if (special) {
        int nCol2D = (int)std::sqrt((float)nproc);
        int rankInRow2D = myrank / nCol2D;
        int rankInCol2D = myrank % nCol2D;
        int sqrtLayer = (int)std::sqrt((float)nlayers);
        rankInFiber = (rankInCol2D % sqrtLayer) * sqrtLayer + (rankInRow2D % sqrtLayer);
        rankInLayer = (rankInRow2D / sqrtLayer) * gridCols + (rankInCol2D / sqrtLayer);
        MPI_Comm_split(world3D, rankInFiber, rankInLayer, &layerWorld);
        MPI_Comm_split(world3D, rankInLayer, rankInFiber, &fiberWorld);
    } else {
        rankInFiber = myrank / procPerLayer;
        rankInLayer = myrank % procPerLayer;
        MPI_Comm_split(world3D, rankInFiber, rankInLayer, &layerWorld);
        MPI_Comm_split(world3D, rankInLayer, rankInFiber, &fiberWorld);
    }

    commGridLayer.reset(new CommGrid(layerWorld, gridRows, gridCols));
}

CommGrid3D::~CommGrid3D()
{
    MPI_Comm_free(&world3D);
    MPI_Comm_free(&fiberWorld);
    MPI_Comm_free(&layerWorld);
}

int CommGrid3D::GetRank(int layerrank, int rowrank, int colrank)
{
    if (!special) return layerrank * gridRows * gridCols + rowrank * gridCols + colrank;
    // Special case logic is not provided.
    return -1;
}

int CommGrid3D::GetGridLayers() { return gridLayers; }
int CommGrid3D::GetGridRows() { return gridRows; }
int CommGrid3D::GetGridCols() { return gridCols; }
int CommGrid3D::GetSize() { return gridLayers * gridRows * gridCols; }
bool CommGrid3D::isSpecial() { return special; }
MPI_Comm& CommGrid3D::GetWorld() { return world3D; }
MPI_Comm& CommGrid3D::GetFiberWorld() { return fiberWorld; }
MPI_Comm& CommGrid3D::GetLayerWorld() { return layerWorld; }
std::shared_ptr<CommGrid> CommGrid3D::GetCommGridLayer() { return commGridLayer; }
int CommGrid3D::GetRankInWorld() { return myrank; }
int CommGrid3D::GetRankInFiber() { return rankInFiber; }
int CommGrid3D::GetRankInLayer() { return rankInLayer; }

}  // namespace combblas