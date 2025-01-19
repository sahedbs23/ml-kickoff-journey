import  numpy as np

# np.full(1,9)
# np_arr = np.arange(10)
# print(np_arr)

# np.random.seed(2)
# rand_arr = np.random.rand(2,4)
# print(rand_arr)
#
# print(np.random.randint(low=1, high=99, size=(2,5)))
# np.log1p([])


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 25, 431],
    [453, 31, 86],
]

X = np.array(X)
y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000].fill

train_linear_regression(X, y)