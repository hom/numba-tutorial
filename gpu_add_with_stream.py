from numba import cuda
import numpy as np
import math
from time import time

@cuda.jit
def vector_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        result[idx] = a[idx] + b[idx]

def main():
    n = 20000 * 10000
    x = np.random.uniform(10, 20, n)
    y = np.random.uniform(10, 20, n)

    start = time()
    # 使用默认流：0
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    z_device = cuda.device_array(n)
    z_streams_device = cuda.device_array(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)

    # kernel
    vector_add[blocks_per_grid, threads_per_block](x_device, y_device, z_device, n)

    # device to host
    default_stream_result = z_device.copy_to_host()
    cuda.synchronize()
    print('gpu vector add time: ' + str(time() - start))

    start = time()
    # 使用 5 个流
    number_of_stream = 5
    # 每个流处理的数据为原来的 1/5
    # 符号 // 得到一个整数结果
    segment_size = n // number_of_stream

    # 创造 5 个 cuda stream
    stream_list = list()
    for i in range(0, number_of_stream):
        stream = cuda.stream()
        stream_list.append(stream)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(segment_size / threads_per_block)
    streams_result = np.empty(n)

    for i in range(0, number_of_stream):
        begin = i * segment_size
        end = begin + segment_size
        x_i_device = cuda.to_device(x[begin : end], stream=stream_list[i])
        y_i_device = cuda.to_device(y[begin : end], stream=stream_list[i])

        # kernel
        vector_add[blocks_per_grid, threads_per_block, stream_list[i]](
            x_i_device,
            y_i_device,
            z_streams_device[begin : end],
            segment_size
        )
        # Device to host
        streams_result[begin : end] = z_streams_device[begin : end].copy_to_host(stream=stream_list[i])
    
    cuda.synchronize()
    print('gpu streams vector add time: ' + str(time() - start))

    if (np.array_equal(default_stream_result, streams_result)):
        print('result correct')

if __name__ == '__main__':
    main()
    