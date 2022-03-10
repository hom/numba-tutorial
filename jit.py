from numba import jit
import numpy as np

SIZE = 20000
x = np.random.random((SIZE, SIZE))

@jit
def print_tan_sum(a):
    sum = 0
    for i in range(SIZE):
        for j in range(SIZE):
            sum += np.tanh(a[i, j])

    return sum

print(print_tan_sum(x))