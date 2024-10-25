import numpy as np

a = [3, 4, 4, 6, 3]
b = [2, 3, 2, 3, 0]

a = np.array(a, dtype=float)
b = np.array(b, dtype=float)

a -= np.mean(a)
b -= np.mean(b)

X = np.array([a, b], dtype=float)

X = X.T
print(X)

cov = X.T @ X
e_vals,e_vecs = np.linalg.eig(cov)
print(e_vals,e_vecs)