import numpy as np
cimport numpy as np
from libc.stdio cimport printf

ctypedef np.int32_t int32_or_int64


cdef _node_degrees(
        np.ndarray[int32_or_int64, ndim=1, mode="c"] ind,
        np.ndarray[int32_or_int64, ndim=1, mode="c"] ptr,
        np.npy_intp num_rows):
    """
    Find the degree of each node (matrix row) in a graph represented
    by a sparse CSR or CSC matrix.
    """
    cdef np.npy_intp ii, jj
    cdef np.ndarray[int32_or_int64] degree = np.zeros(num_rows, dtype=ind.dtype)
    
    for ii in range(num_rows):
        degree[ii] = ptr[ii + 1] - ptr[ii]
        for jj in range(ptr[ii], ptr[ii + 1]):
            if ind[jj] == ii:
                degree[ii] += 1  # add one if the diagonal is in row ii
                break
    
    printf("Degrees: ")
    for ii in range(num_rows):
        printf("ii %d deg %d \n", ii, degree[ii])
    printf("\n")
    
    return degree


def _reverse_cuthill_mckee(np.ndarray[int32_or_int64, ndim=1, mode="c"] ind,
        np.ndarray[int32_or_int64, ndim=1, mode="c"] ptr,
        np.npy_intp num_rows):
    """
    Reverse Cuthill-McKee ordering of a sparse symmetric CSR or CSC matrix.
    """
    cdef np.npy_intp N = 0, N_old, level_start, level_end, temp
    cdef np.npy_intp zz, ii, jj, kk, ll, level_len
    cdef np.ndarray[int32_or_int64] order = np.ones(num_rows, dtype=ind.dtype) * -1
    cdef np.ndarray[int32_or_int64] degree = _node_degrees(ind, ptr, num_rows)
    cdef np.ndarray[np.npy_intp] inds = np.argsort(degree,kind="stable")
    cdef np.ndarray[np.npy_intp] rev_inds = np.argsort(inds,kind="stable")
    cdef np.ndarray[int32_or_int64] temp_degrees = np.zeros(np.max(degree), dtype=ind.dtype)
    cdef int32_or_int64 i, j, seed, temp2
    cdef int32_or_int64 tmpii = 0

    printf("Initial inds: ")
    for ii in range(num_rows):
        printf("%d ", inds[ii])
    printf("\n")

    # loop over zz takes into account possible disconnected graph.
    printf("num_rows %d \n",num_rows)
    for zz in range(num_rows):
        if inds[zz] != -1:
            seed = inds[zz]
            printf("zz %d seed %d \n", zz, seed)
            order[N] = seed
            N += 1
            inds[rev_inds[seed]] = -1
            level_start = N - 1
            level_end = N

            while level_start < level_end:
                printf("current level start : %d level end %d \n", level_start, level_end)
                for ii in range(level_start, level_end):
                    i = order[ii]
                    N_old = N
                    printf("current start node %d \n", i)
                    # add unvisited neighbors
                    for jj in range(ptr[i], ptr[i + 1]):
                        j = ind[jj]
                        if inds[rev_inds[j]] != -1:
                            inds[rev_inds[j]] = -1
                            order[N] = j
                            printf("adding %d to current level\n", j)
                            tmpii += 1
                            N += 1

                    # Add values to temp_degrees array for insertion sort
                    level_len = 0
                    for kk in range(N_old, N):
                        temp_degrees[level_len] = degree[order[kk]]
                        level_len += 1
                    
                    # Do insertion sort for nodes from lowest to highest degree
                    for kk in range(1, level_len):
                        temp = temp_degrees[kk]
                        temp2 = order[N_old+kk]
                        ll = kk
                        while (ll > 0) and (temp < temp_degrees[ll-1]):
                            temp_degrees[ll] = temp_degrees[ll-1]
                            order[N_old+ll] = order[N_old+ll-1]
                            ll -= 1
                        temp_degrees[ll] = temp
                        order[N_old+ll] = temp2
                    printf("N_old %d final level seq: ", N_old)
                    for kk in range(num_rows):
                        printf("%d ", order[kk])
                    printf("finish inserting\n")
                level_start = level_end
                level_end = N

        if N == num_rows:
            break
    tmp = order[::-1]
    printf("adding %d to the next level", tmpii)
    printf("Final order: ")
    for ii in range(num_rows):
        printf("%d ", order[num_rows - ii - 1])
    printf("\n")
    for ii in tmp:
        printf("%d ", degree[ii])
    printf("\n")
    return order[::-1]
