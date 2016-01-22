import numpy as np
import random
import neurolab as nl


def f(x):
    return np.array([x[0] ** x[1]])


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
    n = 1000
    m = 4
    p = 2
    learn_in, learn_out = make_set(n, p, f)
    test_in, test_out = make_set(m, p, f)


    net = nl.net.newff([[0, 1], [0, 1]], [5, 1])
    net.train_gd(learn_in, learn_out)

    print("a", test_in)
    print("b", test_out)
    valid_out = net.sim(test_in)
    print("v", valid_out)




