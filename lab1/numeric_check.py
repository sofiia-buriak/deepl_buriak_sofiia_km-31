from model import compute_loss

EPS = 1e-6
TOL = 1e-7

CHECKED = [
    ("W1", (0, 0)),
    ("b1", (0,)),
    ("W2", (0, 0)),
    ("b2", (0,)),
]


def label(name, index):
    return f"{name}[{', '.join(str(i) for i in index)}]"


def numeric_grad(params, X, y, name, index, eps=EPS):
    work = {k: v.copy() for k, v in params.items()}
    original = work[name][index]

    work[name][index] = original + eps
    loss_plus = compute_loss(work, X, y)

    work[name][index] = original - eps
    loss_minus = compute_loss(work, X, y)

    work[name][index] = original
    return (loss_plus - loss_minus) / (2.0 * eps)
