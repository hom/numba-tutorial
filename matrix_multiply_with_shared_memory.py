from numba import cuda
import numpy as np
import math
from time import time

# thread per block
# 每个 block 有 BLOCK_SIZE X BLOCK_SIZE 个元素
BLOCK_SIZE = 16

@cuda.jit
def matmul(A, B, C):
    """矩阵乘法 C = A * B"""
    # numba 提供了更简易的计算方法
    # x, y = cuda.grid(2)
    # 具体计算公式如下
    row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


@cuda.jit
def matmul_shared_memory(A, B, C):
    """使用 Shared Memory 的矩阵乘法 C = A * B"""
    sA = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    sB = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    if row >= C.shape[0] and col >= C.shape[1]:
        # 当 (x, y) 越界时退出
        return
    
    tmp = 0
    # 以一个 BLOCK_SIZE x BLOCK_SIZE 为单位
    for m in range(math.ceil(A.shape[1] / BLOCK_SIZE)):
        sA[tx, ty] = A[row, ty + m * BLOCK_SIZE]
        sB[tx, ty] = B[tx + m * BLOCK_SIZE, col]

        cuda.syncthreads()

        for n in range(BLOCK_SIZE):
            tmp += sA[tx, n] * sB[n, ty]

        cuda.syncthreads()
    C[row, col] = tmp


def main():
    # 初始化矩阵
    M = 6000
    N = 4800
    P = 4000
    # 随机生成的 [M x N] 矩阵
    A = np.random.random((M, N))
    # 随机生成的 [N x P] 矩阵
    B = np.random.random((N, P))

    start = time()
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array((M, P))

    # 执行配置
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid_x = int(math.ceil(A.shape[0] / BLOCK_SIZE))
    blocks_per_grid_y = int(math.ceil(B.shape[1] / BLOCK_SIZE))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # 启动核函数
    start = time()
    matmul[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    print('matmul time: ' + str(time() - start))

    start = time()
    matmul_shared_memory[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    print('matmul with shared memory time: ' + str(time() - start))
    C = C_device.copy_to_host()


if __name__ == '__main__':
    main()
