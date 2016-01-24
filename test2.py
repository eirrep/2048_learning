import numpy as np
import pybrain as pb


def f(x):
    v = 1 if x[0] > x[1] else 0
    u = 1 if (x[0] + x[1]) > 1 else 0
    return np.array([v, u])


def make_set(n, p, f):
    x = []
    y = []
    for i in range(n):
        a = np.random.rand(p)
        b = f(a)
        x.append(a)
        y.append(b)
    x = np.array(x)
    y = np.array(y)
    return x, y

if __name__ == "__main__":
    x = 2




