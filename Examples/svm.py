from snapvx import *
import numpy as num
import scipy as spy

import time
from itertools import *
import sys

# solve the network lasso problem consisting of a central node, and a dummy node.
# We maximise the log of the determinant of a semidefinite matrix
lamb = 10
C = 0.1
M = 2 + 1
N_Train = 100
N_Test = 100


def create_dataset(n_train, n_test):
    X_x = []
    Y_y = []

    for k in xrange(n_train):
        if num.random.randn() < 0:
            X_x.append([-1 + num.random.randn() * 0.5, -1 + num.random.randn() * 0.5])
            Y_y.append([-1])
        else:
            X_x.append([1 + num.random.randn() * 0.5, 1 + num.random.randn() * 0.5])
            Y_y.append([1])
    X_x_test = []
    Y_y_test = []

    for k in xrange(n_test):
        if num.random.randn() < 0:
            X_x_test.append([-1 + num.random.randn() * 0.5, -1 + num.random.randn() * 0.5])
            Y_y_test.append([-1])
        else:
            X_x_test.append([1 + num.random.randn() * 0.5, 1 * num.random.randn() * 0.5])
            Y_y_test.append([1])

    return X_x, Y_y, X_x_test, Y_y_test


def create_objectives_constraints(X, Y, C):
    A = []
    p = 1
    np = Y.count([1])
    nn = Y.count([-1])
    n = float(nn) / float(np) * (-1)
    for k in xrange(len(Y)):
        temp = list(X[k])
        if Y[k][0] == 1:
            temp.append(p)
            A.append(temp)

        else:
            temp = [(-1) * v for v in temp]
            temp.append(n)
            A.append(temp)

    A = num.matrix(A)
    M, N = A.shape
    y = num.matrix([[1]] * M)
    x = Variable(N, name="x")
    Y = num.array(Y)
    Epsilon = Variable(M, name="E")  # punishing factor
    objective = sum_entries(square(x)) + C * sum_entries(square(Epsilon))
    # constraints = [numpy.multiply(Y,(1-Epsilon)) <= A*x]
    constraints = [y <= A * x + num.diag(Y) * Epsilon]

    return objective, constraints, x


############## Node 1 ################
X, Y, X_Test, Y_Test = create_dataset(N_Train, N_Test)
obj, cons, w = create_objectives_constraints(X, Y, C)

############### Node 2 ##########################
X1, Y1, X1_Test, Y1_Test = create_dataset(N_Train, N_Test)
obj1, cons1, w1 = create_objectives_constraints(X1, Y1, C)

num.random.seed(1)
gvx = TGraphVX()

# add the two nodes
gvx.AddNode(0, Objective=obj, Constraints=cons)
gvx.AddNode(1, Objective=obj1, Constraints=cons1)
# gvx.AddNode(1)

# add an edge between them with the regularisation penalty
gvx.AddEdge(0, 1, Objective=lamb * sum_entries(norm((w - w1), 1)))
# gvx.AddEdge(0, 1)

# Solve the problem
gvx.Solve(Verbose=True, Rho=1.0, UseADMM=True, MaxIters=200)
gvx.PrintSolution()


