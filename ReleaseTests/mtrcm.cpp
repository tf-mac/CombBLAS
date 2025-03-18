//
// Created by Yuxi Hong on 2/26/25.
//

#include <sys/time.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpCCols.h"
#include "fast_matrix_market/app/Eigen.hpp"

using namespace std;
using namespace Eigen;
using namespace combblas;

double get_time_microseconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64_t ll = static_cast<long long>(tv.tv_sec) * 1000000LL + tv.tv_usec;
    return static_cast<double>(ll) * 1e-6;
}

template <typename Scalar>
void printEigenVector(const std::vector<Scalar>& vec, const std::string name)
{
    // Map the std::vector data to an Eigen vector.
    if (!name.empty()) std::cout << name << ": ";
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> eigenVec(vec.data(), vec.size());
    std::cout << eigenVec.transpose() << std::endl;
}

template <class IT, class NT>
IT compute_bandwidth(SpCCols<IT, NT>& spmat)
{
    auto* cscptr = spmat.GetInternal();
    auto cols = cscptr->n;
    auto* ptr = cscptr->jc;
    auto* ind = cscptr->ir;
    IT bandwidth = 0;
    for (IT i = 0; i < cols; ++i) {
        for (IT j = ptr[i]; j < ptr[i + 1]; ++j) {
            bandwidth = std::max(bandwidth, std::abs(ind[j] - i));
        }
    }
    return bandwidth + 1;
}

template <class IT, class NT>
std::vector<IT> computedegree(SpCCols<IT, NT>& spmat)
{
    auto* cscptr = spmat.GetInternal();
    auto cols = cscptr->n;
    auto* ptr = cscptr->jc;
    auto* ind = cscptr->ir;
    std::vector<IT> degree(cols);
    for (IT i = 0; i < cols; ++i) {
        // Basic degree: number of nonzeros in row i
        degree[i] = ptr[i + 1] - ptr[i];
        for (IT j = ptr[i]; j < ptr[i + 1]; ++j) {
            if (ind[j] == i) {
                ++degree[i];
                break;
            }
        }
    }
    return degree;
}

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const { return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1); }
};

template <class IT, class NT>
void blockdesc(SpCCols<IT, NT>& spmat, IT blocksize)
{
    std::cerr << "Block size " << blocksize << std::endl;
    auto* cscptr = spmat.GetInternal();
    auto cols = cscptr->n;
    auto* ptr = cscptr->jc;
    auto* ind = cscptr->ir;
    std::vector<IT> blockrowid = std::vector<IT>(cscptr->nz);
    std::vector<IT> blockcolid = std::vector<IT>(cscptr->nz);
    std::vector<std::pair<IT, IT>> coordvec(cscptr->nz);
    for (IT i = 0; i < cols; i++) {
        for (IT j = ptr[i]; j < ptr[i + 1]; j++) {
            coordvec[j] = std::make_pair(i / blocksize, ind[j] / blocksize);
        }
    }
    std::unordered_map<std::pair<int, int>, int, PairHash> count_map;
    // Count occurrences
    for (const auto& p : coordvec) {
        count_map[p]++;
    }
    // Output the counts
    std::cerr << "Block number " << count_map.size() << std::endl;
    // for (const auto& entry : count_map) {
    //     std::cout << "(" << entry.first.first << ", " << entry.first.second << ") -> " << entry.second << std::endl;
    // }
}

//!< compute the rcm order
template <class IT, class NT>
std::vector<IT> RCM(SpCCols<IT, NT>& spmat)
{
    auto* cscptr = spmat.GetInternal();
    auto cols = cscptr->n;
    auto* ptr = cscptr->jc;
    auto* ind = cscptr->ir;
    std::vector<IT> degree = computedegree(spmat);
    // printEigenVector(degree, "degree");
    std::vector<IT> sorted_nodes(cols, 0);
    std::vector<IT> order(cols, 0);
    std::iota(sorted_nodes.begin(), sorted_nodes.end(), 0);
    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
              [&degree](int a, int b) { return (degree[a] == degree[b]) ? (a < b) : (degree[a] < degree[b]); });
    // printEigenVector(sorted_nodes, "inds");
    // std::vector<IT> rsorted_nodes(sorted_nodes);
    // std::sort(rsorted_nodes.begin(), rsorted_nodes.end(),
    //           [&sorted_nodes](int a, int b) { return (sorted_nodes[a] == sorted_nodes[b]) ? (a < b) : (sorted_nodes[a] < sorted_nodes[b]); });
    // printEigenVector(rsorted_nodes, "rsorted_nodes");
    bool* visited = new bool[cols]();
    for (IT i = 0; i < cols; ++i) visited[i] = false;
    IT N = 0, N_old = 0;
    for (int seed : sorted_nodes) {
        if (!visited[seed]) {
            // printf("starting seed %d \n", seed);
            std::vector<IT> current_level;
            current_level.push_back(seed);
            visited[seed] = true;
            order[N] = seed;
            N++;
            auto level_start = N - 1;
            auto level_end = N;
            while (level_start < level_end) {
                // Process all nodes in the current level.
                for (IT ni = level_start; ni < level_end; ni++) {
                    IT node = order[ni];
                    std::vector<IT> next_level;
                    N_old = N;
                    // For each neighbor of 'node'
                    for (IT j = ptr[node]; j < ptr[node + 1]; ++j) {
                        IT neighbor = ind[j];
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            next_level.push_back(neighbor);
                            N++;  // add 1 elments to the order set
                            // std::cerr << "adding " << neighbor << " to the next level" << std::endl;
                        }
                    }
                    // Sort the nodes in the next level by increasing degree.
                    std::sort(next_level.begin(), next_level.end(),
                              [&degree](int a, int b) { return (degree[a] == degree[b]) ? (a < b) : (degree[a] < degree[b]); });
                    // for (auto node : next_level) {
                    //     printf("node %d deg %d \n", node, degree[node]);
                    // }
                    // user insertion sort?
                    // Append the sorted next level to the overall order.
                    for (IT node : next_level) {
                        order[N_old++] = node;
                    }
                    // printEigenVector(order, "order:");
                }
                level_start = level_end;
                level_end = N;
            }
        }
    }
    std::reverse(order.begin(), order.end());
    delete[] visited;
    return order;
}

template <class IT, class NT>
SpCCols<IT, NT> reorderspmat(SpCCols<IT, NT>& spmat, std::vector<IT> order)
{
    // init local variables
    auto* tuples = new std::tuple<IT, IT, NT>[spmat.getnnz()];  // no need to delete
    IT nidx = 0;
    auto* cscptr = spmat.GetInternal();
    auto cols = cscptr->n;
    auto* ptr = cscptr->jc;
    auto* ind = cscptr->ir;
    auto* val = cscptr->num;
    // std::cerr << "order size " << order.size() << std::endl;
    // get reverseorder
    std::vector<IT> reverseorder(spmat.getncol());
    for (IT i = 0; i < order.size(); ++i) {
        if (order[i] >= cols) {
            std::cerr << "exceed!" << std::endl;
        }
        reverseorder[order[i]] = i;
    }

    for (IT i = 0; i < cols; ++i) {
        for (IT j = ptr[i]; j < ptr[i + 1]; ++j) {
            IT row = ind[j];
            // tuples[nidx] = std::make_tuple(row, i, val[j]);
            if (reverseorder[row] >= cols) {
                std::cerr << "!!!" << std::endl;
            }
            if (reverseorder[i] >= cols) {
                std::cerr << "???" << std::endl;
            }
            tuples[nidx] = std::make_tuple(reverseorder[row], reverseorder[i], val[j]);
            ++nidx;
        }
    }
    auto* spt = new SpTuples<IT, NT>(cscptr->nz, spmat.getnrow(), spmat.getncol(), tuples);
    SpCCols<IT, NT> ret = SpCCols<IT, NT>(*spt, false);
    delete spt;
    return ret;
}

int main()
{
    typedef int32_t IT;
    typedef double NT;

    fast_matrix_market::matrix_market_header header;
    std::vector<IT> rows;
    std::vector<IT> cols;
    std::vector<NT> vals;
    // std::ifstream fhandle("/media/volume/workspace/datasets/ex1/ex1.mtx");
    // std::ifstream fhandle("/media/volume/workspace/datasets/webbase-1M/webbase-1M.mtx");
    // std::ifstream fhandle("/media/volume/workspace/datasets/delaunay_n22/delaunay_n22.mtx");
    std::ifstream fhandle("/media/volume/workspace/datasets/pdb1HYS/pdb1HYS.mtx");
    // std::ifstream fhandle("test323csym.mtx");

    // Eigen::SparseMatrix<double> mat;
    // fast_matrix_market::read_matrix_market_eigen(fhandle, mat);
    // Eigen::IOFormat fmt(2, 0, " ", "\n", "[", "]");
    // std::cout << MatrixXd(mat).format(fmt) << std::endl;
    double t1, t2;
    t1 = get_time_microseconds();
    fast_matrix_market::read_matrix_market_triplet(fhandle, header, rows, cols, vals);
    t2 = get_time_microseconds();
    printf("reading matrix time %.3f \n", t2 - t1);
    auto* tuples = new std::tuple<IT, IT, NT>[rows.size()];
    t1 = get_time_microseconds();
#pragma omp parallel
    for (IT i = 0; i < rows.size(); i++) {
        tuples[i] = std::make_tuple(rows[i], cols[i], vals[i]);
    }
    t2 = get_time_microseconds();
    printf("create tuples time %.3f \n", t2 - t1);
    t1 = get_time_microseconds();
    auto* ptuples = new SpTuples<IT, NT>(rows.size(), header.nrows, header.ncols, tuples);
    SpCCols<IT, NT> spmat = SpCCols<IT, NT>(*ptuples, false);
    t2 = get_time_microseconds();
    printf("sort tuples and create spccols time %.3f \n", t2 - t1);
    t1 = get_time_microseconds();
    auto rcmorder = RCM(spmat);
    t2 = get_time_microseconds();
    printf("RCM time %.3f \n", t2 - t1);
    // printEigenVector(rcmorder, "RCM Order");
    auto degree = computedegree(spmat);
    // for (auto node : rcmorder) {
    //     printf("%d ", degree[node]);
    // }
    // printf("\n");
    // auto rcmorder_sorted = rcmorder;
    // std::sort(rcmorder_sorted.begin(), rcmorder_sorted.end());
    // for (IT i = 0; i < rcmorder_sorted.size(); i++) {
    //     if (rcmorder_sorted[i] != i) {
    //         std::cerr << "not correct sequence!" << std::endl;
    //     }
    // }
    blockdesc(spmat, 128);
    SpCCols<IT, NT> rcmspmat = reorderspmat(spmat, rcmorder);
    blockdesc(rcmspmat, 128);
    // printEigenVector(rcmorder, "RCMORDER");
    // auto degree = computedegree(spmat);
    // auto ncols = spmat.getncol();

    // for (int i = 0; i < 10; i++) std::cerr << rcmorder[ncols - i - 1] << ",";
    // std::cerr << std::endl;
    // for (int i = 0; i < 10; i++) std::cerr << degree[rcmorder[ncols - i - 1]] << ",";
    // std::cerr << std::endl;
    IT bw = compute_bandwidth(spmat);
    std::cerr << "bw " << bw << std::endl;
    IT bw2 = compute_bandwidth(rcmspmat);
    std::cerr << "bw " << bw << " bw2: " << bw2 << std::endl;
    // // delete rcmspmat;
    // delete ptuples;
    return 0;
}