from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_array, csc_array
from fast_matrix_market import mmread

a = mmread("/Users/hongy0a/Documents/dataset/webbase-1M/webbase-1M.mtx")
print(a.nnz)
