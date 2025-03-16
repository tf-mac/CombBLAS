import scipy.sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_array, csc_array
from fast_matrix_market import mmread,mmwrite
import sys
import os
import numpy as np

DLOC = os.getenv("DLOC")
if DLOC == None:
    print("Error DLOC not set.")
    exit(1)


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


# get datasetname
# dname = sys.argv[1]
# fpath = "{}/{}/{}.mtx".format(DLOC, dname, dname)
# if not os.path.exists(fpath):
#     print("fpath {} not exists!".format(fpath))
fpath = "/media/volume/workspace/kk/tfcombblas-minor2/build/test.mtx"
print("reading {}".format(fpath))
a = mmread(fpath).tocsc()
asymm = a
asymm += asymm.transpose()
mmwrite("asym.mtx", asymm)
print(type(a))
rcmorder = reverse_cuthill_mckee(a)
print(a.shape)
degree = np.zeros(a.shape[1])
for i in range(0, a.shape[1]):
    degree[i] = a.indptr[i + 1] - a.indptr[i]
    for j in range(a.indptr[i], a.indptr[i + 1]):
        if a.indices[j] == i:
            degree[i] += 1
            break
cmorder = rcmorder[::-1]
print("Rcm order ", rcmorder)
print(cmorder[:10])
for i in range(10):
    print(degree[cmorder[i]])

a_rcm = a[rcmorder, :][:, rcmorder]
bw = compute_bandwidth(a)
bw2 = compute_bandwidth(a_rcm)
print("bw {} bw2 {}".format( bw, bw2))
