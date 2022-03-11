from numba import cuda
import numpy as np
import math
from time import time

@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if (idx < n):
        result[idx] = a[idx] + b[idx]

def main():
    n = 2000 * 10000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    # 拷贝数据到设备端
    x_to_device = cuda.to_device(x)
    y_to_device = cuda.to_device(y)
    # 在显卡中初始化一块存储 GPU 计算结果的空间
    gpu_result = cuda.device_array(n)
    cpu_result = np.empty(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x_to_device, y_to_device, gpu_result, n)
    cuda.synchronize()
    print('gpu vector add time: ' + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print('cpu vector add time: ' + str(time() - start))

    if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
        print('result correct')

if __name__ == '__main__':
    main()
