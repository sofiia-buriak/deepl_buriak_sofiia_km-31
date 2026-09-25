import numpy as np
import torch
import torch.nn as nn


def build_model(params):
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    ).double()

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(params["W1"].T))
        model[0].bias.copy_(torch.from_numpy(params["b1"]))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T))
        model[2].bias.copy_(torch.from_numpy(params["b2"]))
    return model


def loss_and_grads(params, X, y):
    model = build_model(params)
    X_t = torch.from_numpy(np.ascontiguousarray(X)).double()
    y_t = torch.from_numpy(np.ascontiguousarray(y)).long()

    logits = model(X_t)
    loss = nn.functional.cross_entropy(logits, y_t)
    model.zero_grad()
    loss.backward()

    grads = {
        "W1": model[0].weight.grad.numpy().T.copy(),
        "b1": model[0].bias.grad.numpy().copy(),
        "W2": model[2].weight.grad.numpy().T.copy(),
        "b2": model[2].bias.grad.numpy().copy(),
    }
    return loss.item(), grads
