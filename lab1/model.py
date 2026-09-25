import numpy as np

N_IN, N_HIDDEN, N_OUT = 4, 8, 3
SEED = 0


def init_params(seed=SEED):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / N_IN), size=(N_IN, N_HIDDEN))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (N_HIDDEN + N_OUT)), size=(N_HIDDEN, N_OUT))
    return {
        "W1": W1,
        "b1": np.zeros(N_HIDDEN),
        "W2": W2,
        "b2": np.zeros(N_OUT),
    }


def log_softmax(Z):
    m = Z.max(axis=1, keepdims=True)
    S = Z - m
    return S - np.log(np.exp(S).sum(axis=1, keepdims=True))


def forward(params, X):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ params["W2"] + params["b2"]
    cache = {"X": X, "Z1": Z1, "A1": A1}
    return Z2, cache


def cross_entropy(Z2, y):
    log_P = log_softmax(Z2)
    N = y.shape[0]
    loss = -log_P[np.arange(N), y].mean()
    return loss, log_P


def backward(params, cache, log_P, y, divide_by_n=True):
    N = y.shape[0]
    P = np.exp(log_P)
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1.0

    dZ2 = P - Y
    if divide_by_n:
        dZ2 = dZ2 / N

    dW2 = cache["A1"].T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)

    dW1 = cache["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def compute_loss(params, X, y):
    Z2, _ = forward(params, X)
    loss, _ = cross_entropy(Z2, y)
    return loss


def loss_and_grads(params, X, y, divide_by_n=True):
    Z2, cache = forward(params, X)
    loss, log_P = cross_entropy(Z2, y)
    grads = backward(params, cache, log_P, y, divide_by_n=divide_by_n)
    return loss, grads
