import numpy as np
from rcm import _reverse_cuthill_mckee
from fast_matrix_market import mmread,mmwrite



def compute_bandwidth(A):
    """
    Compute the bandwidth of a sparse matrix in CSC format.

    The bandwidth is defined as:
        bandwidth = max(|i - j| for each nonzero A[i, j]) + 1

    Parameters
    ----------
    A : scipy.sparse.csc_matrix
        Input sparse matrix in CSC format.

    Returns
    -------
    bandwidth : int
        The computed bandwidth.
    """
    max_diff = 0
    # Loop over each column.
    for j in range(A.shape[1]):
        # Loop over all nonzero entries in column j.
        for idx in range(A.indptr[j], A.indptr[j + 1]):
            diff = abs(A.indices[idx] - j)
            if diff > max_diff:
                max_diff = diff
    return max_diff + 1


# # Example CSR representation of a sparse matrix
# ind = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)  # Row indices
# ptr = np.array([0, 2, 4, 6], dtype=np.int32)  # Column pointers
# num_rows = 3
mtx = "test323c.mtx"
# print("Input CSR structure:")
# print("ind:", ind)
# print("ptr:", ptr)
spmat = mmread(mtx).tocsr()
spmat += spmat.transpose()
mmwrite("spmatsym.mtx", spmat)
# Call Reverse Cuthill-McKee function
rcmorder = _reverse_cuthill_mckee(spmat.indices, spmat.indptr, spmat.shape[0])
print("Reverse Cuthill-McKee order:", rcmorder)
spmat_rcm = spmat[rcmorder, :][:, rcmorder]
bw = compute_bandwidth(spmat)
bw2 = compute_bandwidth(spmat_rcm)
print("bw {} bw2 {}".format( bw, bw2))




