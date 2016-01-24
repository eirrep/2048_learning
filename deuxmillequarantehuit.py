from copy import copy
import random
import time
import numpy as np
import sklearn as skl
import neurolab as nl


MAX_SCORE=100
MOVES = ["l", "r", "d", "u"]

class Tableau(object):
    def __init__(self):
        self.values = [0] * 16
        self.score = 0
        self.moves = 0

    def clone(self):
        x = Tableau()
        x.score = self.score
        x.moves = self.moves
        x.values = copy(self.values)
        return x

    def compress(self, line):
        old_line = [x for x in line if x !=0]
        new_line = [0] * len(old_line)
        i = 0
        score = 0
        while i < len(old_line):
            if i + 1 < len(old_line) and old_line[i+1] == 0:
                new_line[i] = 0
                new_line[i+1] = old_line[i]
                i += 2
            elif i + 1 < len(old_line) and old_line[i+1] == old_line[i]:
                new_line[i] = 0
                new_line[i+1] = 1 + old_line[i]
                score += new_line[i+1]
                i += 2
            else:
                new_line[i] = old_line[i]
                i += 1
        self.score += score
        new_line = [x for x in new_line if x != 0]
        new_line += [0] * (4 - len(new_line))
        return new_line

    def add_random(self):
        v = 1 if random.random() < 0.75 else 2
        index_0 = [i for i, x in enumerate(self.values) if x == 0]
        if len(index_0) == 0:
            return False
        index = random.choice(index_0)
        self.values[index] = v
        return True

    def get_lines(self):
        lines = []
        for i in range(4):
            new_line = self.values[4*i:4*(i+1)]
            lines.append(new_line)
        return lines

    def get_columns(self):
        columns = []
        for i in range(4):
            new_col = [self.values[i], self.values[i+4], self.values[i+8], self.values[i+12]]
            columns.append(new_col)
        return columns

    def from_lines(self, lines):
        self.values = []
        for line in lines:
            self.values += line

    def from_columns(self, cols):
        for i, col in enumerate(cols):
            for j in range(4):
                self.values[i + 4*j] = col[j]


    def move(self, dir):
        if dir == 'l':
            v = self.get_lines()
        if dir == 'r':
            v = self.get_lines()
            for x in v:
                x.reverse()
        if dir == 'u':
            v = self.get_columns()
        if dir == 'd':
            v = self.get_columns()
            for x in v:
                x.reverse()
        new_v = [self.compress(x) for x in v]
        if(new_v == v):
            return False
        if dir == 'l':
            self.from_lines(new_v)
        if dir == 'r':
            for x in new_v:
                x.reverse()
            self.from_lines(new_v)
        if dir == 'u':
            self.from_columns(new_v)
        if dir == 'd':
            for x in new_v:
                x.reverse()
            self.from_columns(new_v)
        self.moves += 1
        return True

    def affiche(self):
        s = "----------\n"
        for l in self.get_lines():
            s += "|"
            for x in l:
                if x == 0:
                    s += " "
                else:
                    s += str(x)
                s += " "
            s += "|\n"
        s += "----------"
        print(s)

    def best_from_min_max(self, x):
        t = time.clock()
        depth_max = 3
        moves_score = {}
        for m in MOVES:
            a = self.clone()
            a.move(m)
            moves_score[m] = a.compute_all_random_apparition(1, depth_max)
        moves = [m for m in moves_score if moves_score[m] == max(moves_score.values())]
        t = time.clock() - t
        #print("Hope at move ", self.moves + depth_max, " : ", max(moves_score.values()), "in ", t , " s.")
        return random.choice(moves)

    def compute_all_random_apparition(self, depth, depth_max):
        num_0 = [i for i, x in enumerate(self.values) if x == 0]
        if len(num_0) == 0:
            return -1
        worst_score = None
        random.shuffle(num_0)
        for index in num_0:
            for new_v in [1]:
                new_t = self.clone()
                new_t.values[index] = new_v
                v = new_t.play_all(depth+1, depth_max)
                if worst_score is None or v < worst_score:
                    worst_score = v
        return worst_score

    def play_all(self, depth, depth_max):
        best_score = -1000
        for m in MOVES:
            a = self.clone()
            a.move(m)
            if a.values != self.values:
                if any([x == 0 for x in a.values]) > 0:
                    v = a.score if depth == depth_max else a.compute_all_random_apparition(depth, depth_max)
                    if v > best_score:
                        best_score = v
        return best_score

    def max_next(self, x):
        moves_score = {}
        for m in MOVES:
            a = self.clone()
            a.move(m)
            moves_score[m] = a.score
        moves = [m for m in moves_score if moves_score[m] == max(moves_score.values())]
        print(max(moves_score.values()))
        return random.choice(moves)

    def ask(self, x):
        x = input("Quel mouvement ?")
        if x == "1":
            return 'l'
        if x == "2":
            return 'd'
        if x == "3":
            return 'r'
        if x == "5":
            return 'u'
        return self.ask(x)

    def random_move(self, x):
        return random.choice(MOVES)

    def random_1(self, x):
        if x == 0:
            return "r"
        if x == 1:
            return "d"
        if x == 2:
            return "l"
        return "u"

def play(t, method, aff=False):
    while t.add_random():
        if aff or t.moves % 10 == 0:
            t.affiche()
            print("moves ", t.moves, "   score ", t.score)
        i = 0
        while True:
            m = method(i)
            i += 1
            if t.move(m):
                break
            if i == 4:
                break
    t.affiche()
    print("moves ", t.moves, "   score ", t.score)

def move_to_array(dir):
    if dir == 'l':
        return [1,0,0,0]
    if dir == 'r':
        return [0,1,0,0]
    if dir == 'd':
        return [0,0,1,0]
    if dir == 'u':
        return [0,0,0,1]

def make_set_learning(n):
    input = []
    output = []
    for i in range(n):
        t = Tableau()
        while t.add_random():
            i = 0
            m = t.random_move(i)
            new_input = t.values
            t.move(m)
            new_output = t.score/MAX_SCORE
            if new_output > 1:
                raise ValueError("Test output bigger than one: {}".format(new_output))
            input.append(np.array(new_input))
            output.append(np.array(new_output))
    return input, output

def make_input_size(p, mini, maxi):
    a = [mini, maxi]
    v = [a for i in range(p)]
    return v

def compute_error(output_try, output_true):
    ecart = np.sqrt(np.mean((output_try - output_true)**2))
    return ecart

def nn():
    n = 100
    m = 4
    p = 20
    learn_in, learn_out = make_set_learning(n)
    test_in, test_out = make_set_learning(m)
    print("Set computed")
    input_size = [[0,15] for x in range(16)]

    net = nl.net.newff(input_size, [32, 1])
    net.trainf = nl.train.train_rprop
    ecart_array = []
    while True:
        err = net.train(learn_in, learn_out, goal=-0.01, epochs=10, show=10)
        test_out_nn = net.sim(test_in)
        ecart = compute_error(test_out, test_out_nn)
        ecart_array.append(ecart)
        print(ecart_array)
        if len(ecart_array) > 2 and ecart_array[-1] > ecart_array[-2]*0.98:
            break


if __name__ == '__main__':
    nn()
