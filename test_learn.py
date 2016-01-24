import numpy as np
import neurolab as nl
import time



def f(x):
    u = 0
    v = 0
    for i, a in enumerate(x):
        u += a
        v += (-1)**i * a
    return np.array([u/len(x), v])


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

def make_input_size(p, mini, maxi):
    a = [mini, maxi]
    v = [a for i in range(p)]
    return v


def nn():
    n = 1000
    m = 4
    p = 50
    learn_in, learn_out = make_set(n, p, f)
    test_in, test_out = make_set(m, p, f)
    input_size = make_input_size(p, 0, 1)


    net = nl.net.newff(input_size, [10, 10, 2])
    net.trainf = nl.train.train_rprop
    err = net.train(learn_in, learn_out, goal=-0.01, epochs=500, show=50)

    print("a", test_in)
    print("b", test_out)
    valid_out = net.sim(test_in)
    print("v", valid_out)
    ecart = valid_out - test_out
    print("error", ecart)
    print("mean error", np.sqrt(np.mean(ecart**2)))

if __name__ == "__main__":
    t = time.clock()
    nn()
    t = time.clock() - t
    print("Done in {} seconds.".format(t))