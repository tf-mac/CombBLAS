//  Project AC-SpGEMM
//  https://www.tugraz.at/institute/icg/research/team-steinberger/
//
//  Copyright (C) 2018 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Martin Winter - martin.winter (at) icg.tugraz.at
//              Daniel Mlakar - daniel.mlakar (at) icg.tugraz.at
//              Rhaleb Zayer - rzayer (at) mpi-inf.mpg.de
//              Hans-Peter Seidel - hpseidel (at) mpi-inf.mpg.de
//              Markus Steinberger - steinberger ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

#pragma once

#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <parallel/algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fmm = fast_matrix_market;

#include "Vector.h"

template <typename T>
struct COO {
    size_t rows, cols, nnz;

    std::unique_ptr<T[]> data;
    std::unique_ptr<unsigned int[]> row_ids;
    std::unique_ptr<unsigned int[]> col_ids;

    COO() : rows(0), cols(0), nnz(0) {}
    void alloc(size_t rows, size_t cols, size_t nnz);
    void sorted(bool columnfirst);
    void saveMTX(std::string filename);
};

namespace
{
struct DataTypeValidator {
    static bool validate(std::string type) { return false; }
};
/*
    template<>
    struct DataTypeValidator<float> {
        static const bool validate(std::string type) {
            return type.compare("real") == 0 || type.compare("integer") == 0;
        }
    };
    template<typename VALUE_TYPE>
    struct DataTypeValidator  {
        static const bool validate(std::string type) {
           std::cout << "type: " << type << std::endl;
            return type.compare("real") == 0 || type.compare("integer") == 0;;
        }
    };

    template<>
    struct DataTypeValidator<int> {
        static const bool validate(std::string type) {
            return type.compare("integer") == 0;
        }
    };

    template<>
    struct DataTypeValidator<uint64_t> {
        static const bool validate(std::string type) {
            return type.compare("integer") == 0;
        }
    };*/
}  // namespace

template <typename T>
void COO<T>::alloc(size_t r, size_t c, size_t n)
{
    rows = r;
    cols = c;
    nnz = n;

    data = std::make_unique<T[]>(n);
    row_ids = std::make_unique<unsigned int[]>(n);
    col_ids = std::make_unique<unsigned int[]>(n);
}
template <typename T>
void COO<T>::sorted(bool columnfirst)
{
    size_t size = nnz;
    // Step 1: Create index array
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = i;
    }

    if (columnfirst) {
        // Step 2: Sort indices based on col_ids and row_ids
        //        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        //            if (col_ids[a] != col_ids[b]) return col_ids[a] < col_ids[b];
        //            return row_ids[a] < row_ids[b];
        //        });
        __gnu_parallel::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            if (col_ids[a] != col_ids[b]) return col_ids[a] < col_ids[b];
            return row_ids[a] < row_ids[b];
        });
    } else {
        // Step 2: Sort indices based on row_ids and col_ids
        //        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        //            if (row_ids[a] != row_ids[b]) return row_ids[a] < row_ids[b];
        //            return col_ids[a] < col_ids[b];
        //        });
        __gnu_parallel::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            if (row_ids[a] != row_ids[b]) return row_ids[a] < row_ids[b];
            return col_ids[a] < col_ids[b];
        });
    }

    // Step 3: Apply permutation to all arrays
    std::unique_ptr<T[]> new_data(new T[size]);
    std::unique_ptr<unsigned int[]> new_row_ids(new unsigned int[size]);
    std::unique_ptr<unsigned int[]> new_col_ids(new unsigned int[size]);

    for (size_t i = 0; i < size; ++i) {
        new_data[i] = data[indices[i]];
        new_row_ids[i] = row_ids[indices[i]];
        new_col_ids[i] = col_ids[indices[i]];
    }

    // Step 4: Replace original arrays
    data = std::move(new_data);
    row_ids = std::move(new_row_ids);
    col_ids = std::move(new_col_ids);
}
template <typename T>
void COO<T>::saveMTX(std::string filename)
{
    // Open output file stream
    std::ofstream fout(filename);
    // std::cerr << "start writing!" << std::endl;
    // Write to Matrix Market using fastMatrixMarket
    std::vector<uint32_t> row_indices(row_ids.get(), row_ids.get() + nnz),
        col_indices(col_ids.get(), col_ids.get() + nnz);
    std::vector<T> datavec(data.get(), data.get() + nnz);
    // std::cerr << "rowindice size:" << row_indices.size() << std::endl;
    // std::cerr << "row col" << rows << "," << cols << std::endl;
    fmm::matrix_market_header header;
    header.nrows = rows;
    header.ncols = cols;
    header.nnz = nnz;
    fmm::write_matrix_market_triplet(fout, header, row_indices, col_indices, datavec);
    fout.close();
}

template <typename T>
COO<T> loadMTX(const char* file)
{
    std::ifstream fstream(file);
    if (!fstream.is_open()) throw std::runtime_error(std::string("could not open \"") + file + "\"");
    std::vector<uint32_t> row_indices, col_indices;
    std::vector<T> datavec;
    fmm::matrix_market_header header;
    fmm::read_matrix_market_triplet(fstream, header, row_indices, col_indices, datavec);
    COO<T> resmatrix;
    resmatrix.alloc(header.nrows, header.ncols, header.nnz);
    resmatrix.row_ids = std::make_unique<unsigned int[]>(header.nnz);
    resmatrix.col_ids = std::make_unique<unsigned int[]>(header.nnz);
    resmatrix.data = std::make_unique<T[]>(header.nnz);
    std::copy(row_indices.begin(), row_indices.end(), resmatrix.row_ids.get());
    std::copy(col_indices.begin(), col_indices.end(), resmatrix.col_ids.get());
    std::copy(datavec.begin(), datavec.end(), resmatrix.data.get());
    return resmatrix;
}

template <typename T>
COO<T> loadCOO(const char* file)
{
    return COO<T>();
}

template <typename T>
void storeCOO(const COO<T>& mat, const char* file)
{
}

template <typename T>
void spmv(DenseVector<T>& res, const COO<T>& m, const DenseVector<T>& v, bool transpose)
{
    if (transpose && v.size != m.rows) throw std::runtime_error("SPMV dimensions mismatch");
    if (!transpose && v.size != m.cols) throw std::runtime_error("SPMV dimensions mismatch");

    size_t outsize = transpose ? m.cols : m.rows;
    if (res.size < outsize) res.data = std::make_unique<T[]>(outsize);
    res.size = outsize;

    std::fill(&res.data[0], &res.data[0] + outsize, 0);

    if (transpose)
        for (size_t i = 0; i < m.nnz; ++i) res.data[m.col_ids[i]] += m.data[i] * v.data[m.row_ids[i]];
    else
        for (size_t i = 0; i < m.nnz; ++i) res.data[m.row_ids[i]] += m.data[i] * v.data[m.col_ids[i]];
}
