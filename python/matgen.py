import random

# Set a fixed seed for reproducibility (remove or change for different random values)
random.seed(42)

n = 24
nnz_per_col = 10

# Total number of nonzeros: 64 columns * 10 nnz per column = 640.
total_nnz = n * nnz_per_col

# Print the Matrix Market header and size information.
print("%%MatrixMarket matrix coordinate real general")
print("% 64 x 64 random sparse matrix with 10 nonzeros per column (640 nonzeros total)")
print(f"{n} {n} {total_nnz}")

# For each column, choose 10 unique random row indices and assign a random value.
for col in range(1, n+1):
    # random.sample returns 10 unique rows (in the range 1..64)
    rows = random.sample(range(1, n+1), nnz_per_col)
    rows.sort()  # Sorting rows is optional but makes the file easier to read.
    for row in rows:
        # Generate a random float value (in [0,1)).
        val = random.random()
        # Each nonzero entry is written as: row col value
        print(f"{row} {col} {val:.6f}")
