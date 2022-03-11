from numba import cuda

def cpu_print():
    print('printed by cpu')

@cuda.jit
def gpu_print():
    print('printed by gpu')

def main():
    gpu_print[2, 4]()
    cuda.synchronize()
    cpu_print()

if __name__ == '__main__':
    main()