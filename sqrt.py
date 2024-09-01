import random
import numpy as np

def sqrt(n, thr=0.001):
    x = random.randint(2, n)
    while True:
        x -= 1.e-6 * (4 * (x ** 3) - 4 * n * x)
        if x ** 2 - n < 0.001:
            break
    return x

if __name__ == "__main__":
    print(sqrt(2))
